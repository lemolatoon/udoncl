#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "udoncl/udon_abi_layout.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/RegionGraphTraits.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <chrono>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <tuple>

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_PROPAGATEDEVICERESOURCEPROVENANCEPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {

constexpr StringLiteral discoveryAttribute =
    "vrc.static_device_resource_discovery";
constexpr StringLiteral provenanceAttribute =
    "vrc.static_device_resource_provenance";
constexpr StringLiteral allocationSiteAttribute =
    "vrc.static_device_allocation_site";
constexpr StringLiteral callSiteAttribute = "vrc.static_device_call_site";
constexpr StringLiteral operandAttribute =
    "vrc.static_device_operand_provenance";
constexpr StringLiteral resultAttribute = "vrc.static_device_result_provenance";
constexpr StringLiteral argumentAttribute =
    "vrc.static_device_argument_provenance";
constexpr StringLiteral functionResultAttribute =
    "vrc.static_device_function_result_provenance";
constexpr StringLiteral resourceOperandContractAttribute =
    "vrc.device_resource_operands";
constexpr StringLiteral bufferViewLayoutAttribute = "vrc.buffer_view_layout";
constexpr StringLiteral debugProvenanceAttribute =
    "vrc.debug_device_resource_provenance";

struct Fact {
  bool initialized = false;
  bool resource = false;
  bool unknown = false;
  bool mayBeNull = false;
  bool zero = false;
  SmallVector<int32_t> ids;

  static Fact scalar() {
    Fact fact;
    fact.initialized = true;
    return fact;
  }
  static Fact nullHandle() {
    Fact fact;
    fact.initialized = true;
    fact.resource = true;
    fact.mayBeNull = true;
    return fact;
  }
  static Fact zeroScalar() {
    Fact fact = scalar();
    fact.zero = true;
    return fact;
  }
  static Fact identity(int32_t id) {
    Fact fact;
    fact.initialized = true;
    fact.resource = true;
    fact.ids.push_back(id);
    return fact;
  }
};

Fact joinFacts(Fact left, const Fact &right);
bool operator==(const Fact &left, const Fact &right);
bool operator!=(const Fact &left, const Fact &right);

struct CellState {
  bool hasPath = false;
  bool initializedOnAllPaths = false;
  Fact fact;
};

using CellMap = llvm::DenseMap<Value, CellState>;
using ObjectId = uint32_t;

struct ObjectRef {
  ObjectId base = 0;
  int32_t offset = 0;

  bool operator<(const ObjectRef &other) const {
    return std::tie(base, offset) < std::tie(other.base, other.offset);
  }

  bool operator==(const ObjectRef &other) const {
    return base == other.base && offset == other.offset;
  }
};

struct BufferViewLayout {
  uint32_t count = 0;
  uint32_t stride = 0;
  SmallVector<int32_t> memberOffsets;
};

template <typename Key, typename Value> class CopyOnWriteMap {
  using Storage = std::map<Key, Value>;
  std::shared_ptr<Storage> storage = std::make_shared<Storage>();

  void ensureUnique() {
    if (!storage.unique())
      storage = std::make_shared<Storage>(*storage);
  }

public:
  using const_iterator = typename Storage::const_iterator;

  bool empty() const { return storage->empty(); }
  size_t size() const { return storage->size(); }
  size_t count(const Key &key) const { return storage->count(key); }
  const_iterator find(const Key &key) const { return storage->find(key); }
  const_iterator find(const Key &key) { return storage->find(key); }
  const_iterator begin() const { return storage->begin(); }
  const_iterator end() const { return storage->end(); }
  const_iterator begin() { return storage->begin(); }
  const_iterator end() { return storage->end(); }
  const Value &at(const Key &key) const { return storage->at(key); }
  Value lookup(const Key &key) const {
    auto found = storage->find(key);
    return found == storage->end() ? Value() : found->second;
  }

  Value &operator[](const Key &key) {
    ensureUnique();
    return (*storage)[key];
  }

  template <typename... Args> auto try_emplace(const Key &key, Args &&...args) {
    ensureUnique();
    return storage->try_emplace(key, std::forward<Args>(args)...);
  }

  size_t erase(const Key &key) {
    ensureUnique();
    return storage->erase(key);
  }

  template <typename Predicate> void eraseIf(Predicate predicate) {
    if (llvm::none_of(*storage, [&](const auto &entry) {
          return predicate(entry.first, entry.second);
        }))
      return;
    ensureUnique();
    for (auto iterator = storage->begin(); iterator != storage->end();)
      if (predicate(iterator->first, iterator->second))
        iterator = storage->erase(iterator);
      else
        ++iterator;
  }

  void clear() {
    if (storage->empty())
      return;
    storage = std::make_shared<Storage>();
  }

  Storage &mutableStorage() {
    ensureUnique();
    return *storage;
  }

  friend bool operator==(const CopyOnWriteMap &left,
                         const CopyOnWriteMap &right) {
    return left.storage == right.storage || *left.storage == *right.storage;
  }

  friend bool operator!=(const CopyOnWriteMap &left,
                         const CopyOnWriteMap &right) {
    return !(left == right);
  }
};

struct ObjectRefSet {
  bool unknown = false;
  SmallVector<ObjectRef> refs;

  static ObjectRefSet singleton(ObjectRef object) {
    ObjectRefSet result;
    result.refs.push_back(object);
    return result;
  }
};

using ResourceCellMap = CopyOnWriteMap<ObjectRef, CellState>;

enum class HandleFieldKind : uint8_t { Slot, Generation };

struct HandleFieldOrigin {
  ObjectRef handle;
  HandleFieldKind kind = HandleFieldKind::Slot;

  bool operator<(const HandleFieldOrigin &other) const {
    return std::tie(handle, kind) < std::tie(other.handle, other.kind);
  }

  bool operator==(const HandleFieldOrigin &other) const {
    return handle == other.handle && kind == other.kind;
  }
};

struct HandleFieldOriginSet {
  bool unknown = false;
  SmallVector<HandleFieldOrigin> refs;
};

bool operator==(const HandleFieldOriginSet &left,
                const HandleFieldOriginSet &right) {
  return left.unknown == right.unknown && left.refs == right.refs;
}

HandleFieldOriginSet joinHandleFieldOrigins(HandleFieldOriginSet left,
                                            const HandleFieldOriginSet &right) {
  left.unknown |= right.unknown;
  left.refs.append(right.refs);
  llvm::sort(left.refs);
  left.refs.erase(std::unique(left.refs.begin(), left.refs.end()),
                  left.refs.end());
  return left;
}

std::optional<HandleFieldOrigin>
exactHandleFieldOrigin(const HandleFieldOriginSet &origins) {
  if (origins.unknown || origins.refs.size() != 1)
    return std::nullopt;
  return origins.refs.front();
}

bool operator==(const ObjectRefSet &left, const ObjectRefSet &right) {
  return left.unknown == right.unknown && left.refs == right.refs;
}

ObjectRefSet joinObjectRefs(ObjectRefSet left, const ObjectRefSet &right) {
  left.unknown |= right.unknown;
  left.refs.append(right.refs);
  llvm::sort(left.refs);
  left.refs.erase(std::unique(left.refs.begin(), left.refs.end()),
                  left.refs.end());
  return left;
}

std::optional<ObjectRef> exactObject(const ObjectRefSet &objects) {
  if (objects.unknown || objects.refs.size() != 1)
    return std::nullopt;
  return objects.refs.front();
}

using HandleMap = CopyOnWriteMap<ObjectRef, Fact>;

struct ViewState {
  bool hasPath = false;
  bool initialized = false;
  bool mayBeInvalid = false;
  uint8_t directFieldMask = 0;
  HandleFieldOriginSet slotOrigins;
  HandleFieldOriginSet generationOrigins;
  Fact resource;
};

bool operator==(const ViewState &left, const ViewState &right) {
  return left.hasPath == right.hasPath &&
         left.initialized == right.initialized &&
         left.mayBeInvalid == right.mayBeInvalid &&
         left.directFieldMask == right.directFieldMask &&
         left.slotOrigins == right.slotOrigins &&
         left.generationOrigins == right.generationOrigins &&
         left.resource == right.resource;
}

bool operator!=(const ViewState &left, const ViewState &right) {
  return !(left == right);
}

using ViewMap = CopyOnWriteMap<ObjectRef, ViewState>;

ViewState joinViewStates(ViewState left, const ViewState &right) {
  if (!left.hasPath)
    return right;
  if (!right.hasPath)
    return left;
  left.initialized &= right.initialized;
  left.mayBeInvalid |= right.mayBeInvalid;
  left.directFieldMask &= right.directFieldMask;
  left.slotOrigins =
      joinHandleFieldOrigins(std::move(left.slotOrigins), right.slotOrigins);
  left.generationOrigins = joinHandleFieldOrigins(
      std::move(left.generationOrigins), right.generationOrigins);
  left.resource = joinFacts(std::move(left.resource), right.resource);
  return left;
}

ViewMap joinViewMaps(const ViewMap &left, const ViewMap &right) {
  ViewMap joined;
  for (const auto &[object, state] : left) {
    auto found = right.find(object);
    ViewState other = found == right.end() ? ViewState{true} : found->second;
    joined[object] = joinViewStates(state, other);
  }
  for (const auto &[object, state] : right) {
    if (left.count(object))
      continue;
    joined[object] = joinViewStates(ViewState{true}, state);
  }
  return joined;
}

struct PointerCellState {
  bool hasPath = false;
  bool initializedOnAllPaths = false;
  ObjectRefSet objects;
};

using PointerCellMap = CopyOnWriteMap<ObjectRef, PointerCellState>;

template <typename Map>
SmallVector<ObjectRef> changedObjectRefs(const Map &previous, const Map &next) {
  SmallVector<ObjectRef> changed;
  auto left = previous.begin();
  auto right = next.begin();
  while (left != previous.end() || right != next.end()) {
    if (right == next.end() ||
        (left != previous.end() && left->first < right->first)) {
      changed.push_back(left->first);
      ++left;
      continue;
    }
    if (left == previous.end() || right->first < left->first) {
      changed.push_back(right->first);
      ++right;
      continue;
    }
    if (!(left->second == right->second))
      changed.push_back(left->first);
    ++left;
    ++right;
  }
  return changed;
}

using StatusExclusions = std::set<int64_t>;
using StatusConstraintMap = llvm::DenseMap<Value, StatusExclusions>;

struct UnknownPointerExitCase {
  StatusExclusions excluded;
  PointerCellMap pointerCells;
};

struct PointerExitCases {
  std::map<int64_t, PointerCellMap> exact;
  SmallVector<UnknownPointerExitCase> unknown;
};

struct UnknownViewExitCase {
  StatusExclusions excluded;
  ViewMap views;
};

struct ViewExitCases {
  std::map<int64_t, ViewMap> exact;
  SmallVector<UnknownViewExitCase> unknown;
};

struct UnknownResourceCellExitCase {
  StatusExclusions excluded;
  ResourceCellMap cells;
};

struct ResourceCellExitCases {
  std::map<int64_t, ResourceCellMap> exact;
  SmallVector<UnknownResourceCellExitCase> unknown;
};

bool operator==(const PointerCellState &left, const PointerCellState &right) {
  return left.hasPath == right.hasPath &&
         left.initializedOnAllPaths == right.initializedOnAllPaths &&
         left.objects == right.objects;
}

bool operator!=(const PointerCellState &left, const PointerCellState &right) {
  return !(left == right);
}

struct AnalysisResult {
  SmallVector<Fact> results;
  SmallVector<std::optional<int64_t>> integerResults;
  SmallVector<std::optional<ObjectRefSet>> objectResults;
  SmallVector<HandleFieldOriginSet> handleFieldResults;
  HandleMap handles;
  ViewMap views;
  ResourceCellMap resourceCells;
  PointerCellMap pointerCells;
  ResourceCellExitCases resourceCellExitCases;
  PointerExitCases pointerExitCases;
  ViewExitCases viewExitCases;
};

PointerCellState joinPointerCells(PointerCellState left,
                                  const PointerCellState &right) {
  if (!left.hasPath)
    return right;
  if (!right.hasPath)
    return left;
  left.initializedOnAllPaths &= right.initializedOnAllPaths;
  left.objects = joinObjectRefs(std::move(left.objects), right.objects);
  return left;
}

PointerCellMap joinPointerMaps(const PointerCellMap &left,
                               const PointerCellMap &right,
                               const llvm::DenseSet<ObjectId> &zeroObjects) {
  auto missingState = [&](ObjectRef object) {
    return zeroObjects.contains(object.base)
               ? PointerCellState{true, true, {}}
               : PointerCellState{true, false, {}};
  };
  PointerCellMap joined;
  for (const auto &[object, state] : left) {
    auto found = right.find(object);
    PointerCellState other =
        found == right.end() ? missingState(object) : found->second;
    joined[object] = joinPointerCells(state, other);
  }
  for (const auto &[object, state] : right) {
    if (left.count(object))
      continue;
    joined[object] = joinPointerCells(missingState(object), state);
  }
  return joined;
}

void addPointerExitCase(PointerExitCases &cases, std::optional<int64_t> status,
                        const PointerCellMap &pointerCells,
                        const llvm::DenseSet<ObjectId> &zeroObjects,
                        const StatusExclusions &excluded = {}) {
  constexpr size_t maxExactCases = 16;
  if (!status ||
      (!cases.exact.count(*status) && cases.exact.size() >= maxExactCases)) {
    auto found = llvm::find_if(cases.unknown, [&](const auto &candidate) {
      return candidate.excluded == excluded;
    });
    if (found == cases.unknown.end())
      cases.unknown.push_back({excluded, pointerCells});
    else
      found->pointerCells =
          joinPointerMaps(found->pointerCells, pointerCells, zeroObjects);
    return;
  }
  auto found = cases.exact.find(*status);
  if (found == cases.exact.end())
    cases.exact.emplace(*status, pointerCells);
  else
    found->second = joinPointerMaps(found->second, pointerCells, zeroObjects);
}

PointerCellMap
selectPointerExitCases(const PointerExitCases &cases, int64_t constant,
                       bool equal, const PointerCellMap &fallback,
                       const llvm::DenseSet<ObjectId> &zeroObjects) {
  PointerCellMap selected;
  bool hasSelected = false;
  for (const auto &[status, pointerCells] : cases.exact) {
    if ((status == constant) != equal)
      continue;
    selected = hasSelected
                   ? joinPointerMaps(selected, pointerCells, zeroObjects)
                   : pointerCells;
    hasSelected = true;
  }
  for (const UnknownPointerExitCase &unknown : cases.unknown) {
    if (equal && unknown.excluded.count(constant))
      continue;
    selected = hasSelected ? joinPointerMaps(selected, unknown.pointerCells,
                                             zeroObjects)
                           : unknown.pointerCells;
    hasSelected = true;
  }
  return hasSelected ? selected : fallback;
}

void addViewExitCase(ViewExitCases &cases, std::optional<int64_t> status,
                     const ViewMap &views,
                     const StatusExclusions &excluded = {}) {
  constexpr size_t maxExactCases = 16;
  if (!status ||
      (!cases.exact.count(*status) && cases.exact.size() >= maxExactCases)) {
    auto found = llvm::find_if(cases.unknown, [&](const auto &candidate) {
      return candidate.excluded == excluded;
    });
    if (found == cases.unknown.end())
      cases.unknown.push_back({excluded, views});
    else
      found->views = joinViewMaps(found->views, views);
    return;
  }
  auto found = cases.exact.find(*status);
  if (found == cases.exact.end())
    cases.exact.emplace(*status, views);
  else
    found->second = joinViewMaps(found->second, views);
}

ViewMap selectViewExitCases(const ViewExitCases &cases, int64_t constant,
                            bool equal, const ViewMap &fallback) {
  ViewMap selected;
  bool hasSelected = false;
  for (const auto &[status, views] : cases.exact) {
    if ((status == constant) != equal)
      continue;
    selected = hasSelected ? joinViewMaps(selected, views) : views;
    hasSelected = true;
  }
  for (const UnknownViewExitCase &unknown : cases.unknown) {
    if (equal && unknown.excluded.count(constant))
      continue;
    selected =
        hasSelected ? joinViewMaps(selected, unknown.views) : unknown.views;
    hasSelected = true;
  }
  return hasSelected ? selected : fallback;
}

template <typename ExitCases>
bool hasMatchingExitCase(const ExitCases &cases, int64_t constant, bool equal) {
  for (const auto &[status, state] : cases.exact)
    if ((status == constant) == equal)
      return true;
  for (const auto &unknown : cases.unknown)
    if (!equal || !unknown.excluded.count(constant))
      return true;
  return false;
}

template <typename Combine>
std::optional<int64_t> foldExactInteger(Type type, int64_t left, int64_t right,
                                        Combine combine) {
  auto integer = dyn_cast<IntegerType>(type);
  if (!integer || integer.getWidth() > 64)
    return std::nullopt;
  llvm::APInt leftBits(integer.getWidth(), static_cast<uint64_t>(left), true);
  llvm::APInt rightBits(integer.getWidth(), static_cast<uint64_t>(right), true);
  return combine(leftBits, rightBits).getSExtValue();
}

std::optional<int64_t>
foldExactIntegerComparison(Type type, int64_t left, int64_t right,
                           arith::CmpIPredicate predicate) {
  auto integer = dyn_cast<IntegerType>(type);
  if (!integer || integer.getWidth() > 64)
    return std::nullopt;
  llvm::APInt leftBits(integer.getWidth(), static_cast<uint64_t>(left), true);
  llvm::APInt rightBits(integer.getWidth(), static_cast<uint64_t>(right), true);
  bool result = false;
  switch (predicate) {
  case arith::CmpIPredicate::eq:
    result = leftBits.eq(rightBits);
    break;
  case arith::CmpIPredicate::ne:
    result = leftBits.ne(rightBits);
    break;
  case arith::CmpIPredicate::slt:
    result = leftBits.slt(rightBits);
    break;
  case arith::CmpIPredicate::sle:
    result = leftBits.sle(rightBits);
    break;
  case arith::CmpIPredicate::sgt:
    result = leftBits.sgt(rightBits);
    break;
  case arith::CmpIPredicate::sge:
    result = leftBits.sge(rightBits);
    break;
  case arith::CmpIPredicate::ult:
    result = leftBits.ult(rightBits);
    break;
  case arith::CmpIPredicate::ule:
    result = leftBits.ule(rightBits);
    break;
  case arith::CmpIPredicate::ugt:
    result = leftBits.ugt(rightBits);
    break;
  case arith::CmpIPredicate::uge:
    result = leftBits.uge(rightBits);
    break;
  }
  return result ? 1 : 0;
}

void mergeExactInteger(Value value, std::optional<int64_t> incoming,
                       llvm::DenseMap<Value, int64_t> &integerValues,
                       llvm::DenseSet<Value> &ambiguousIntegers) {
  if (ambiguousIntegers.contains(value))
    return;
  auto existing = integerValues.find(value);
  if (!incoming) {
    if (existing != integerValues.end()) {
      integerValues.erase(existing);
      ambiguousIntegers.insert(value);
    }
    return;
  }
  if (existing == integerValues.end()) {
    integerValues[value] = *incoming;
    return;
  }
  if (existing->second != *incoming) {
    integerValues.erase(existing);
    ambiguousIntegers.insert(value);
  }
}

bool operator==(const Fact &left, const Fact &right) {
  return left.initialized == right.initialized &&
         left.resource == right.resource && left.unknown == right.unknown &&
         left.mayBeNull == right.mayBeNull && left.zero == right.zero &&
         left.ids == right.ids;
}

bool operator!=(const Fact &left, const Fact &right) {
  return !(left == right);
}

Fact joinFacts(Fact left, const Fact &right) {
  if (!left.initialized)
    return right;
  if (!right.initialized)
    return left;
  if (left.resource != right.resource) {
    bool scalarIsZero = left.resource ? right.zero : left.zero;
    if (scalarIsZero) {
      left.resource = true;
      left.mayBeNull = true;
    } else {
      left.resource = true;
      left.unknown = true;
    }
  }
  left.unknown |= right.unknown;
  left.mayBeNull |= right.mayBeNull;
  left.zero &= right.zero;
  left.ids.append(right.ids);
  llvm::sort(left.ids);
  left.ids.erase(std::unique(left.ids.begin(), left.ids.end()), left.ids.end());
  return left;
}

bool operator==(const CellState &left, const CellState &right) {
  return left.hasPath == right.hasPath &&
         left.initializedOnAllPaths == right.initializedOnAllPaths &&
         left.fact == right.fact;
}

bool operator!=(const CellState &left, const CellState &right) {
  return !(left == right);
}

CellState joinCellState(const CellState &left, const CellState &right) {
  if (!left.hasPath)
    return right;
  if (!right.hasPath)
    return left;
  CellState joined;
  joined.hasPath = true;
  joined.initializedOnAllPaths =
      left.initializedOnAllPaths && right.initializedOnAllPaths;
  joined.fact = joinFacts(left.fact, right.fact);
  return joined;
}

ResourceCellMap joinResourceCellMaps(const ResourceCellMap &left,
                                     const ResourceCellMap &right) {
  ResourceCellMap joined;
  auto missing = [] { return CellState{true, false, Fact()}; };
  for (const auto &[cell, state] : left) {
    auto found = right.find(cell);
    joined[cell] =
        joinCellState(state, found == right.end() ? missing() : found->second);
  }
  for (const auto &[cell, state] : right) {
    if (!left.count(cell))
      joined[cell] = joinCellState(missing(), state);
  }
  return joined;
}

void addResourceCellExitCase(ResourceCellExitCases &cases,
                             std::optional<int64_t> status,
                             const ResourceCellMap &cells,
                             const StatusExclusions &excluded = {}) {
  constexpr size_t maxExactCases = 16;
  if (!status ||
      (!cases.exact.count(*status) && cases.exact.size() >= maxExactCases)) {
    auto found = llvm::find_if(cases.unknown, [&](const auto &candidate) {
      return candidate.excluded == excluded;
    });
    if (found == cases.unknown.end())
      cases.unknown.push_back({excluded, cells});
    else
      found->cells = joinResourceCellMaps(found->cells, cells);
    return;
  }
  auto found = cases.exact.find(*status);
  if (found == cases.exact.end())
    cases.exact.emplace(*status, cells);
  else
    found->second = joinResourceCellMaps(found->second, cells);
}

ResourceCellMap selectResourceCellExitCases(const ResourceCellExitCases &cases,
                                            int64_t constant, bool equal,
                                            const ResourceCellMap &fallback) {
  ResourceCellMap selected;
  bool hasSelected = false;
  for (const auto &[status, cells] : cases.exact) {
    if ((status == constant) != equal)
      continue;
    selected = hasSelected ? joinResourceCellMaps(selected, cells) : cells;
    hasSelected = true;
  }
  for (const UnknownResourceCellExitCase &unknown : cases.unknown) {
    if (equal && unknown.excluded.count(constant))
      continue;
    selected = hasSelected ? joinResourceCellMaps(selected, unknown.cells)
                           : unknown.cells;
    hasSelected = true;
  }
  return hasSelected ? selected : fallback;
}

struct Invocation {
  FlatSymbolRefAttr root;
  ArrayAttr path;
};

struct PendingMemoryEffect {
  Operation *call = nullptr;
  ResourceCellExitCases resourceCellCases;
  PointerExitCases pointerCases;
  ViewExitCases viewCases;
};

PendingMemoryEffect transformPendingComparison(
    Operation *operation, const PendingMemoryEffect &source, int64_t constant,
    bool equal, const llvm::DenseSet<ObjectId> &zeroObjects) {
  PendingMemoryEffect transformed;
  transformed.call = operation;
  auto resultFor = [&](int64_t status) {
    return static_cast<int64_t>((status == constant) == equal);
  };
  for (const auto &[status, cells] : source.resourceCellCases.exact)
    addResourceCellExitCase(transformed.resourceCellCases, resultFor(status),
                            cells);
  for (const auto &unknown : source.resourceCellCases.unknown) {
    if (unknown.excluded.count(constant)) {
      addResourceCellExitCase(transformed.resourceCellCases, equal ? 0 : 1,
                              unknown.cells);
    } else {
      addResourceCellExitCase(transformed.resourceCellCases, 0, unknown.cells);
      addResourceCellExitCase(transformed.resourceCellCases, 1, unknown.cells);
    }
  }
  for (const auto &[status, cells] : source.pointerCases.exact)
    addPointerExitCase(transformed.pointerCases, resultFor(status), cells,
                       zeroObjects);
  for (const auto &unknown : source.pointerCases.unknown) {
    if (unknown.excluded.count(constant)) {
      addPointerExitCase(transformed.pointerCases, equal ? 0 : 1,
                         unknown.pointerCells, zeroObjects);
    } else {
      addPointerExitCase(transformed.pointerCases, 0, unknown.pointerCells,
                         zeroObjects);
      addPointerExitCase(transformed.pointerCases, 1, unknown.pointerCells,
                         zeroObjects);
    }
  }
  for (const auto &[status, views] : source.viewCases.exact)
    addViewExitCase(transformed.viewCases, resultFor(status), views);
  for (const auto &unknown : source.viewCases.unknown) {
    if (unknown.excluded.count(constant)) {
      addViewExitCase(transformed.viewCases, equal ? 0 : 1, unknown.views);
    } else {
      addViewExitCase(transformed.viewCases, 0, unknown.views);
      addViewExitCase(transformed.viewCases, 1, unknown.views);
    }
  }
  return transformed;
}

struct ComparisonRefinement {
  Value subject;
  int64_t constant = 0;
  bool equalOnTrue = true;
};

Value successorOperand(Block *predecessor, Block *destination, unsigned index) {
  Operation *terminator = predecessor->getTerminator();
  if (auto branch = dyn_cast<cf::BranchOp>(terminator))
    return branch.getDest() == destination &&
                   index < branch.getDestOperands().size()
               ? branch.getDestOperands()[index]
               : Value();
  if (auto branch = dyn_cast<cf::CondBranchOp>(terminator)) {
    if (branch.getTrueDest() == destination &&
        index < branch.getTrueDestOperands().size())
      return branch.getTrueDestOperands()[index];
    if (branch.getFalseDest() == destination &&
        index < branch.getFalseDestOperands().size())
      return branch.getFalseDestOperands()[index];
    return {};
  }
  if (auto branch = dyn_cast<cf::SwitchOp>(terminator)) {
    if (branch.getDefaultDestination() == destination &&
        index < branch.getDefaultOperands().size())
      return branch.getDefaultOperands()[index];
    for (auto [caseIndex, successor] :
         llvm::enumerate(branch.getCaseDestinations()))
      if (successor == destination &&
          index < branch.getCaseOperands(caseIndex).size())
        return branch.getCaseOperands(caseIndex)[index];
  }
  return {};
}

struct Case {
  Invocation invocation;
  unsigned index;
  Fact fact;
};

using CaseMap = llvm::DenseMap<Operation *, SmallVector<Case>>;

bool sameInvocation(const Invocation &left, const Invocation &right) {
  return left.root == right.root && left.path == right.path;
}

void appendCase(CaseMap &map, Operation *operation,
                const Invocation &invocation, unsigned index,
                const Fact &fact) {
  if (!fact.resource)
    return;
  auto &cases = map[operation];
  for (Case &entry : cases)
    if (entry.index == index && sameInvocation(entry.invocation, invocation)) {
      entry.fact = joinFacts(entry.fact, fact);
      return;
    }
  cases.push_back({invocation, index, fact});
}

bool isAllocation(StringRef name) {
  return name == "clCreateBuffer" || name == "udonclCreateBufferForQueue";
}

bool isKnownOpenCLConsumer(StringRef name) {
  return name == "clReleaseMemObject" || name == "clEnqueueWriteBuffer" ||
         name == "clEnqueueReadBuffer" || name == "clEnqueueCopyBuffer" ||
         name == "clEnqueueFillBuffer" || name == "clSetKernelArg";
}

bool isConstantI32(Value value, int32_t expected) {
  auto constant = value.getDefiningOp<arith::ConstantIntOp>();
  return constant && constant.value() == expected;
}

bool hasAlignment(Operation *operation, int64_t expected) {
  auto alignment = operation->getAttrOfType<IntegerAttr>("alignment");
  return alignment && alignment.getInt() == expected;
}

bool isHandleType(Type type) {
  auto pointer = dyn_cast<::vrc::udonhl::PtrType>(type);
  return pointer && pointer.getAddressSpace() == 0;
}

bool isExactCellAccess(OpOperand &use, Value cell) {
  Operation *operation = use.getOwner();
  if (auto load = dyn_cast<::vrc::udonhl::LoadOp>(operation))
    return use.getOperandNumber() == 0 && load.getPointer() == cell &&
           isHandleType(load.getType()) && hasAlignment(load, 4) &&
           !load->hasAttr("volatile") && !load->hasAttr("volatile_");
  if (auto store = dyn_cast<::vrc::udonhl::StoreOp>(operation))
    return use.getOperandNumber() == 1 && store.getPointer() == cell &&
           isHandleType(store.getValue().getType()) && hasAlignment(store, 4) &&
           !store->hasAttr("volatile") && !store->hasAttr("volatile_");
  return false;
}

bool isSingletonCell(::vrc::udonhl::AllocaOp allocation) {
  auto pointer = dyn_cast<::vrc::udonhl::PtrType>(allocation.getType());
  return pointer && pointer.getAddressSpace() == 0 &&
         isConstantI32(allocation.getByteSize(), 4) &&
         hasAlignment(allocation, 4) &&
         llvm::all_of(allocation.getResult().getUses(), [&](OpOperand &use) {
           if (isExactCellAccess(use, allocation.getResult()))
             return true;
           auto call = dyn_cast<func::CallOp>(use.getOwner());
           return call &&
                  (call.getCallee() == "udonclDeviceBufferGetRaw" ||
                   call.getCallee() == "udonclBufferViewResolve") &&
                  use.getOperandNumber() == 1;
         });
}

bool isHandleObject(::vrc::udonhl::AllocaOp allocation) {
  auto pointer = dyn_cast<::vrc::udonhl::PtrType>(allocation.getType());
  auto alignment = allocation->getAttrOfType<IntegerAttr>("alignment");
  return pointer && pointer.getAddressSpace() == 0 &&
         isConstantI32(allocation.getByteSize(), 8) && alignment &&
         alignment.getInt() >= 4;
}

std::optional<uint32_t>
constantAllocationBytes(::vrc::udonhl::AllocaOp allocation) {
  auto constant =
      allocation.getByteSize().getDefiningOp<arith::ConstantIntOp>();
  if (!constant || constant.value() < 0 ||
      constant.value() > std::numeric_limits<uint32_t>::max())
    return std::nullopt;
  return static_cast<uint32_t>(constant.value());
}

FailureOr<uint64_t> fixedObjectBytes(const DataLayout &layout, Type type) {
  if (isa<LLVM::LLVMPointerType, ::vrc::udonhl::PtrType>(type))
    return uint64_t{4};
  if (auto array = dyn_cast<LLVM::LLVMArrayType>(type)) {
    auto elementBytes = fixedObjectBytes(layout, array.getElementType());
    if (failed(elementBytes) ||
        array.getNumElements() > std::numeric_limits<uint32_t>::max() /
                                     std::max<uint64_t>(*elementBytes, 1))
      return failure();
    return array.getNumElements() * *elementBytes;
  }
  if (auto structure = dyn_cast<LLVM::LLVMStructType>(type)) {
    if (structure.isOpaque())
      return failure();
    uint64_t offset = 0;
    uint64_t aggregateAlignment = structure.isPacked() ? 1 : 0;
    for (Type field : structure.getBody()) {
      auto fieldBytes = fixedObjectBytes(layout, field);
      uint64_t alignment = isa<LLVM::LLVMPointerType>(field)
                               ? 4
                               : layout.getTypeABIAlignment(field);
      if (failed(fieldBytes) || !alignment)
        return failure();
      if (!structure.isPacked()) {
        offset = llvm::alignTo(offset, alignment);
        aggregateAlignment = std::max(aggregateAlignment, alignment);
      }
      if (*fieldBytes > std::numeric_limits<uint32_t>::max() - offset)
        return failure();
      offset += *fieldBytes;
    }
    return llvm::alignTo(offset, aggregateAlignment);
  }
  auto size = layout.getTypeSize(type);
  if (size.isScalable() ||
      size.getFixedValue() > std::numeric_limits<uint32_t>::max())
    return failure();
  return size.getFixedValue();
}

struct PropagateDeviceResourceProvenancePass
    : impl::PropagateDeviceResourceProvenancePassBase<
          PropagateDeviceResourceProvenancePass> {
  ModuleOp module;
  Builder *builder = nullptr;
  ArrayAttr contexts;
  llvm::StringMap<func::FuncOp> functions;
  CaseMap operandCases;
  CaseMap resultCases;
  CaseMap argumentCases;
  CaseMap functionResultCases;
  llvm::DenseSet<Operation *> active;
  llvm::StringMap<::vrc::udonhl::GlobalOp> eligibleGlobals;
  llvm::StringMap<Fact> globalFacts;
  llvm::StringMap<Fact> globalWrites;
  llvm::StringSet<> resourceGlobals;
  llvm::StringMap<::vrc::udonhl::GlobalOp> eligibleHandleGlobals;
  llvm::StringMap<ObjectId> globalObjects;
  llvm::StringMap<ObjectId> globalHandleObjects;
  llvm::DenseSet<ObjectId> globalObjectIds;
  llvm::DenseSet<ObjectId> globalHandleObjectIds;
  llvm::DenseSet<ObjectId> zeroInitializedGlobalObjectIds;
  HandleMap globalHandleFacts;
  HandleMap globalHandleWrites;
  ViewMap globalViewFacts;
  ViewMap globalViewWrites;
  PointerCellMap globalPointerFacts;
  PointerCellMap globalPointerWrites;
  llvm::DenseMap<ObjectId, uint32_t> objectSizes;
  llvm::DenseMap<ObjectId, BufferViewLayout> restrictedBufferViewLayouts;
  std::map<std::tuple<std::string, std::string, Operation *>, ObjectId>
      localHandleObjects;
  ObjectId nextObjectId = 0;
  bool hasConfiguredRoots = false;
  bool collectingGlobalSummaries = false;
  bool debugProgress = false;
  bool invalid = false;
  uint64_t analysisCalls = 0;

  struct AnalysisCacheEntry {
    SmallVector<Fact> arguments;
    SmallVector<std::optional<int64_t>> argumentIntegers;
    SmallVector<std::optional<ObjectRefSet>> argumentObjects;
    SmallVector<HandleFieldOriginSet> argumentHandleFields;
    HandleMap handles;
    ViewMap views;
    ResourceCellMap resourceCells;
    PointerCellMap pointerCells;
    AnalysisResult result;
    bool complete = false;
  };
  std::map<std::string, AnalysisCacheEntry> analysisCache;

  void recordCase(CaseMap &map, Operation *operation,
                  const Invocation &invocation, unsigned index,
                  const Fact &fact) {
    if (!collectingGlobalSummaries)
      appendCase(map, operation, invocation, index, fact);
  }

  template <typename Map>
  static void
  eraseLocalObjectEntries(Map &map,
                          const llvm::DenseSet<ObjectId> &localObjects) {
    map.eraseIf([&](ObjectRef object, const auto &) {
      return localObjects.contains(object.base);
    });
  }

  static void
  eraseLocalObjectEntries(ResourceCellExitCases &cases,
                          const llvm::DenseSet<ObjectId> &localObjects) {
    for (auto &[status, cells] : cases.exact)
      eraseLocalObjectEntries(cells, localObjects);
    for (auto &entry : cases.unknown)
      eraseLocalObjectEntries(entry.cells, localObjects);
  }

  static void
  eraseLocalObjectEntries(PointerExitCases &cases,
                          const llvm::DenseSet<ObjectId> &localObjects) {
    for (auto &[status, cells] : cases.exact)
      eraseLocalObjectEntries(cells, localObjects);
    for (auto &entry : cases.unknown)
      eraseLocalObjectEntries(entry.pointerCells, localObjects);
  }

  static void
  eraseLocalObjectEntries(ViewExitCases &cases,
                          const llvm::DenseSet<ObjectId> &localObjects) {
    for (auto &[status, views] : cases.exact)
      eraseLocalObjectEntries(views, localObjects);
    for (auto &entry : cases.unknown)
      eraseLocalObjectEntries(entry.views, localObjects);
  }

  static bool
  referencesLocalObject(const ObjectRefSet &objects,
                        const llvm::DenseSet<ObjectId> &localObjects) {
    return llvm::any_of(objects.refs, [&](ObjectRef object) {
      return localObjects.contains(object.base);
    });
  }

  static bool
  referencesLocalObject(const HandleFieldOriginSet &origins,
                        const llvm::DenseSet<ObjectId> &localObjects) {
    return llvm::any_of(origins.refs, [&](const HandleFieldOrigin &origin) {
      return localObjects.contains(origin.handle.base);
    });
  }

  static bool
  referencesLocalObject(const PointerCellMap &cells,
                        const llvm::DenseSet<ObjectId> &localObjects) {
    return llvm::any_of(cells, [&](const auto &entry) {
      return !localObjects.contains(entry.first.base) &&
             referencesLocalObject(entry.second.objects, localObjects);
    });
  }

  static bool
  referencesLocalObject(const ViewMap &views,
                        const llvm::DenseSet<ObjectId> &localObjects) {
    return llvm::any_of(views, [&](const auto &entry) {
      return !localObjects.contains(entry.first.base) &&
             (referencesLocalObject(entry.second.slotOrigins, localObjects) ||
              referencesLocalObject(entry.second.generationOrigins,
                                    localObjects));
    });
  }

  static bool
  referencesLocalObject(const PointerExitCases &cases,
                        const llvm::DenseSet<ObjectId> &localObjects) {
    return llvm::any_of(cases.exact,
                        [&](const auto &entry) {
                          return referencesLocalObject(entry.second,
                                                       localObjects);
                        }) ||
           llvm::any_of(cases.unknown, [&](const auto &entry) {
             return referencesLocalObject(entry.pointerCells, localObjects);
           });
  }

  static bool
  referencesLocalObject(const ViewExitCases &cases,
                        const llvm::DenseSet<ObjectId> &localObjects) {
    return llvm::any_of(cases.exact,
                        [&](const auto &entry) {
                          return referencesLocalObject(entry.second,
                                                       localObjects);
                        }) ||
           llvm::any_of(cases.unknown, [&](const auto &entry) {
             return referencesLocalObject(entry.views, localObjects);
           });
  }

  std::string invocationPathKey(ArrayAttr path) {
    std::string storage;
    llvm::raw_string_ostream stream(storage);
    path.print(stream);
    return storage;
  }

  std::string analysisCacheKey(func::FuncOp function,
                               const Invocation &invocation) {
    return (function.getSymName() + "|" + invocation.root.getValue() + "|" +
            invocationPathKey(invocation.path))
        .str();
  }

  static bool
  sameAnalysisInputs(const AnalysisCacheEntry &entry, ArrayRef<Fact> arguments,
                     ArrayRef<std::optional<int64_t>> argumentIntegers,
                     ArrayRef<std::optional<ObjectRefSet>> argumentObjects,
                     ArrayRef<HandleFieldOriginSet> argumentHandleFields,
                     const HandleMap &handles, const ViewMap &views,
                     const ResourceCellMap &resourceCells,
                     const PointerCellMap &pointerCells) {
    return llvm::equal(entry.arguments, arguments) &&
           llvm::equal(entry.argumentIntegers, argumentIntegers) &&
           llvm::equal(entry.argumentObjects, argumentObjects) &&
           llvm::equal(entry.argumentHandleFields, argumentHandleFields) &&
           entry.handles == handles && entry.views == views &&
           entry.resourceCells == resourceCells &&
           entry.pointerCells == pointerCells;
  }

  ObjectId localObject(const Invocation &invocation, Operation *allocation,
                       uint32_t bytes) {
    auto key = std::make_tuple(invocation.root.getValue().str(),
                               invocationPathKey(invocation.path), allocation);
    auto [position, inserted] =
        localHandleObjects.try_emplace(std::move(key), nextObjectId);
    if (inserted)
      ++nextObjectId;
    objectSizes[position->second] = bytes;
    return position->second;
  }

  bool canHoldView(const ObjectRef &object) const {
    auto found = objectSizes.find(object.base);
    if (found == objectSizes.end() || object.offset < 0 ||
        static_cast<uint64_t>(object.offset) +
                UDONCL_UDON_ABI_BUFFER_VIEW_BYTES >
            found->second)
      return false;
    auto restricted = restrictedBufferViewLayouts.find(object.base);
    if (restricted == restrictedBufferViewLayouts.end())
      return true;
    const BufferViewLayout &layout = restricted->second;
    for (int32_t memberOffset : layout.memberOffsets) {
      int64_t delta = static_cast<int64_t>(object.offset) - memberOffset;
      if (delta >= 0 && layout.stride != 0 && delta % layout.stride == 0 &&
          static_cast<uint64_t>(delta / layout.stride) < layout.count)
        return true;
    }
    return false;
  }

  bool canHoldHandle(const ObjectRef &object) const {
    auto found = objectSizes.find(object.base);
    return found != objectSizes.end() && object.offset >= 0 &&
           static_cast<uint64_t>(object.offset) + 8 <= found->second;
  }

  bool isGlobalHandle(const ObjectRef &object) const {
    return globalHandleObjectIds.contains(object.base) && object.offset >= 0 &&
           object.offset % 8 == 0 && canHoldHandle(object);
  }

  bool hasHandle(const HandleMap &handles, const ObjectRef &object) const {
    return handles.count(object) || isGlobalHandle(object);
  }

  Fact lookupHandle(const HandleMap &handles, const ObjectRef &object) const {
    auto found = handles.find(object);
    if (found != handles.end())
      return found->second;
    return isGlobalHandle(object) ? Fact::nullHandle() : Fact();
  }

  bool canHoldResourceCell(const ObjectRef &object) const {
    auto found = objectSizes.find(object.base);
    return found != objectSizes.end() && object.offset >= 0 &&
           object.offset % 4 == 0 &&
           static_cast<uint64_t>(object.offset) + 4 <= found->second;
  }

  bool hasResourceOperandContract(func::CallOp call,
                                  llvm::DenseMap<Value, Fact> &facts) {
    auto found = functions.find(call.getCallee());
    if (found == functions.end() || !found->getValue().isExternal() ||
        call.getNumResults() != 1 || !call.getResult(0).getType().isInteger(32))
      return false;
    auto contract = found->getValue()->getAttrOfType<ArrayAttr>(
        resourceOperandContractAttribute);
    if (!contract)
      return false;
    llvm::DenseSet<unsigned> allowed;
    for (Attribute attribute : contract) {
      auto entry = dyn_cast<DictionaryAttr>(attribute);
      auto index = entry ? entry.getAs<IntegerAttr>("index") : IntegerAttr();
      auto access = entry ? entry.getAs<StringAttr>("access") : StringAttr();
      auto capture = entry ? entry.getAs<StringAttr>("capture") : StringAttr();
      if (!index || index.getInt() < 0 ||
          static_cast<uint64_t>(index.getInt()) >= call.getNumOperands() ||
          !access ||
          (access.getValue() != "read" && access.getValue() != "write" &&
           access.getValue() != "readwrite") ||
          !capture || capture.getValue() != "none" ||
          !allowed.insert(static_cast<unsigned>(index.getInt())).second)
        return false;
    }
    return llvm::all_of(llvm::enumerate(call.getOperands()), [&](auto item) {
      return !valueFact(item.value(), facts).resource ||
             allowed.contains(item.index());
    });
  }

  void completeDirectView(ViewState &state, const HandleMap &handles) {
    constexpr uint8_t allFields = 0x1f;
    state.initialized = false;
    if (state.directFieldMask != allFields)
      return;
    auto slot = exactHandleFieldOrigin(state.slotOrigins);
    auto generation = exactHandleFieldOrigin(state.generationOrigins);
    if (!slot || !generation || slot->kind != HandleFieldKind::Slot ||
        generation->kind != HandleFieldKind::Generation ||
        !(slot->handle == generation->handle))
      return;
    if (!hasHandle(handles, slot->handle))
      return;
    Fact resource = lookupHandle(handles, slot->handle);
    if (!resource.resource || resource.unknown)
      return;
    state.resource = resource;
    state.initialized = true;
    state.mayBeInvalid = true;
  }

  void recordGlobalViewWrite(ObjectRef view, const ViewState &state) {
    if (!collectingGlobalSummaries || !globalObjectIds.contains(view.base) ||
        !state.hasPath || !state.initialized || !canHoldView(view) ||
        !state.resource.resource || state.resource.unknown)
      return;
    ViewState persistent = state;
    // Cross-export summaries describe the resource identity a global view may
    // hold after any export. Concrete lifecycle checks still reject an
    // uninitialized view on the execution path being compiled.
    persistent.initialized = true;
    persistent.mayBeInvalid = true;
    auto filterOrigins = [&](HandleFieldOriginSet &origins) {
      auto firstRemoved =
          std::remove_if(origins.refs.begin(), origins.refs.end(),
                         [&](const HandleFieldOrigin &origin) {
                           return !globalObjectIds.contains(origin.handle.base);
                         });
      if (firstRemoved != origins.refs.end()) {
        origins.refs.erase(firstRemoved, origins.refs.end());
        origins.unknown = true;
      }
    };
    filterOrigins(persistent.slotOrigins);
    filterOrigins(persistent.generationOrigins);
    auto write = globalViewWrites.find(view);
    if (write != globalViewWrites.end())
      persistent = joinViewStates(write->second, persistent);
    globalViewWrites[view] = std::move(persistent);
  }

  bool recordDirectViewField(const ObjectRef &field, Type storedType,
                             const HandleFieldOriginSet *storedOrigins,
                             ViewMap &views, const HandleMap &handles) {
    auto integer = dyn_cast<IntegerType>(storedType);
    if (!integer || integer.getWidth() != 32)
      return false;
    constexpr int32_t offsets[] = {0, 4, 8, 12, 16};
    bool recognized = false;
    for (auto [index, offset] : llvm::enumerate(offsets)) {
      int64_t baseOffset = static_cast<int64_t>(field.offset) - offset;
      if (baseOffset < std::numeric_limits<int32_t>::min() ||
          baseOffset > std::numeric_limits<int32_t>::max())
        continue;
      ObjectRef view{field.base, static_cast<int32_t>(baseOffset)};
      if (!canHoldView(view))
        continue;
      recognized = true;
      ViewState &state = views[view];
      state.hasPath = true;
      state.directFieldMask |= static_cast<uint8_t>(1u << index);
      if (offset == 0)
        state.slotOrigins =
            storedOrigins ? *storedOrigins : HandleFieldOriginSet{true};
      else if (offset == 4)
        state.generationOrigins =
            storedOrigins ? *storedOrigins : HandleFieldOriginSet{true};
      completeDirectView(state, handles);
      recordGlobalViewWrite(view, state);
    }
    return recognized;
  }

  void copyViewSubobjects(Value destination, Value source, Value byteSize,
                          llvm::DenseMap<Value, ObjectRefSet> &objectValues,
                          llvm::DenseMap<Value, int64_t> &integerValues,
                          ViewMap &views) {
    auto destinationObjects = objectValues.find(destination);
    auto sourceObjects = objectValues.find(source);
    auto bytes = integerValues.find(byteSize);
    if (destinationObjects == objectValues.end() ||
        sourceObjects == objectValues.end() || bytes == integerValues.end() ||
        bytes->second <= 0)
      return;
    auto exactDestination = exactObject(destinationObjects->second);
    auto exactSource = exactObject(sourceObjects->second);
    if (!exactDestination || !exactSource)
      return;
    int64_t sourceBegin = exactSource->offset;
    int64_t sourceEnd = sourceBegin + bytes->second;
    int64_t destinationBegin = exactDestination->offset;
    int64_t destinationEnd = destinationBegin + bytes->second;
    SmallVector<std::pair<ObjectRef, ViewState>> copied;
    for (const auto &[view, state] : views) {
      int64_t viewBegin = view.offset;
      int64_t viewEnd = viewBegin + UDONCL_UDON_ABI_BUFFER_VIEW_BYTES;
      if (view.base != exactSource->base || viewBegin < sourceBegin ||
          viewEnd > sourceEnd)
        continue;
      int64_t copiedOffset = destinationBegin + (viewBegin - sourceBegin);
      if (copiedOffset < std::numeric_limits<int32_t>::min() ||
          copiedOffset > std::numeric_limits<int32_t>::max())
        continue;
      ObjectRef copiedView{exactDestination->base,
                           static_cast<int32_t>(copiedOffset)};
      if (canHoldView(copiedView))
        copied.push_back({copiedView, state});
    }
    views.eraseIf([&](ObjectRef view, const ViewState &) {
      int64_t viewBegin = view.offset;
      int64_t viewEnd = viewBegin + UDONCL_UDON_ABI_BUFFER_VIEW_BYTES;
      return view.base == exactDestination->base &&
             viewBegin < destinationEnd && destinationBegin < viewEnd;
    });
    for (auto &[view, state] : copied) {
      views[view] = std::move(state);
      recordGlobalViewWrite(view, views[view]);
    }
  }

  ::vrc::udonhl::GlobalOp globalCell(Value pointer) {
    auto address = pointer.getDefiningOp<::vrc::udonhl::AddressOfOp>();
    if (!address)
      return {};
    auto found = eligibleGlobals.find(address.getGlobalAttr().getValue());
    return found == eligibleGlobals.end() ? ::vrc::udonhl::GlobalOp()
                                          : found->getValue();
  }

  Fact valueFact(Value value, llvm::DenseMap<Value, Fact> &facts) {
    auto found = facts.find(value);
    if (found != facts.end())
      return found->second;
    return Fact::scalar();
  }

  ObjectRefSet
  objectAlternative(Value value, llvm::DenseMap<Value, Fact> &facts,
                    llvm::DenseMap<Value, ObjectRefSet> &objectValues) {
    auto object = objectValues.find(value);
    if (object != objectValues.end())
      return object->second;
    ObjectRefSet alternative;
    if (!valueFact(value, facts).zero)
      alternative.unknown = true;
    return alternative;
  }

  void updateHandle(ObjectRef object, const Fact &incoming, HandleMap &handles,
                    SmallVectorImpl<Block *> &worklist,
                    llvm::DenseSet<Block *> &queued,
                    std::map<ObjectRef, llvm::DenseSet<Block *>> &readers) {
    Fact previous = lookupHandle(handles, object);
    Fact joined = joinFacts(previous, incoming);
    if (previous == joined)
      return;
    handles[object] = joined;
    for (Block *block : readers[object])
      if (queued.insert(block).second)
        worklist.push_back(block);
  }

  void analyzePhysicalBufferResolver(func::CallOp call,
                                     llvm::DenseMap<Value, Fact> &facts) {
    auto found = functions.find(call.getCallee());
    if (found == functions.end() || !found->getValue().isExternal()) {
      call.emitError("udonclResolvePhysicalBuffer requires an unlinked "
                     "external declaration during static resource analysis");
      invalid = true;
      return;
    }
    Type pointer = ::vrc::udonhl::PtrType::get(module.getContext(), 0);
    Type i32 = builder->getI32Type();
    auto expected = builder->getFunctionType({pointer, pointer}, {i32});
    bool callMatches = call.getNumOperands() == 2 &&
                       call.getNumResults() == 1 &&
                       call.getOperand(0).getType() == pointer &&
                       call.getOperand(1).getType() == pointer &&
                       call.getResult(0).getType() == i32;
    if (found->getValue().getFunctionType() != expected || !callMatches) {
      call.emitError("udonclResolvePhysicalBuffer requires signature "
                     "(!udonhl.ptr<0>, !udonhl.ptr<0>) -> i32");
      invalid = true;
      return;
    }
    Fact handle = valueFact(call.getOperand(0), facts);
    if (handle.resource && handle.unknown) {
      call.emitError("resource use has unresolved non-handle provenance in @")
          << call->getParentOfType<func::FuncOp>().getSymName();
      invalid = true;
    }
    if (valueFact(call.getOperand(1), facts).resource) {
      call.emitError("static device resource passed as "
                     "udonclResolvePhysicalBuffer output cell");
      invalid = true;
    }
  }

  std::optional<int32_t> allocationIdentity(const Invocation &invocation,
                                            func::CallOp call) {
    auto site = call->getAttrOfType<DictionaryAttr>(allocationSiteAttribute);
    if (!site) {
      call.emitError(
          "allocation is missing static device allocation-site metadata");
      invalid = true;
      return std::nullopt;
    }
    std::optional<int32_t> identity;
    for (Attribute attribute : contexts) {
      auto context = dyn_cast<DictionaryAttr>(attribute);
      if (!context || context.get("root") != invocation.root ||
          context.get("call_path") != invocation.path ||
          context.get("allocation_site") != site)
        continue;
      auto id = dyn_cast_or_null<IntegerAttr>(context.get("context_id"));
      if (!id)
        continue;
      if (identity) {
        call.emitError(
            "allocation matches multiple static device resource contexts");
        invalid = true;
        return std::nullopt;
      }
      identity = static_cast<int32_t>(id.getInt());
    }
    if (identity)
      return identity;
    call.emitError("allocation has no matching static device resource context");
    invalid = true;
    return std::nullopt;
  }

  Invocation calleeInvocation(const Invocation &caller, func::FuncOp function,
                              func::CallOp call) {
    auto site = call->getAttrOfType<DictionaryAttr>(callSiteAttribute);
    if (!site) {
      call.emitError(
          "internal call is missing static device call-site metadata");
      invalid = true;
      return caller;
    }
    SmallVector<Attribute> path(caller.path.begin(), caller.path.end());
    path.push_back(builder->getDictionaryAttr({
        builder->getNamedAttr(
            "caller",
            FlatSymbolRefAttr::get(module.getContext(), function.getSymName())),
        builder->getNamedAttr("callee", site.get("callee")),
        builder->getNamedAttr("occurrence", site.get("occurrence")),
    }));
    return {caller.root, builder->getArrayAttr(path)};
  }

  AnalysisResult analyze(func::FuncOp function, const Invocation &invocation,
                         ArrayRef<Fact> arguments,
                         ArrayRef<std::optional<int64_t>> argumentIntegers,
                         ArrayRef<std::optional<ObjectRefSet>> argumentObjects,
                         ArrayRef<HandleFieldOriginSet> argumentHandleFields,
                         const HandleMap &incomingHandles,
                         const ViewMap &incomingViews,
                         const ResourceCellMap &incomingResourceCells,
                         const PointerCellMap &incomingPointerCells) {
    ++analysisCalls;
    if (debugProgress && analysisCalls % 10000 == 0)
      llvm::errs() << "udoncl-provenance-analysis=" << analysisCalls
                   << " root=" << invocation.root.getValue()
                   << " function=" << function.getSymName()
                   << " handles=" << incomingHandles.size()
                   << " views=" << incomingViews.size()
                   << " resource_cells=" << incomingResourceCells.size()
                   << " pointer_cells=" << incomingPointerCells.size() << "\n";
    if (!active.insert(function).second) {
      if (llvm::none_of(arguments,
                        [](const Fact &fact) { return fact.resource; }))
        return {SmallVector<Fact>(function.getFunctionType().getNumResults(),
                                  Fact::scalar()),
                SmallVector<std::optional<int64_t>>(
                    function.getFunctionType().getNumResults(), std::nullopt),
                SmallVector<std::optional<ObjectRefSet>>(
                    function.getFunctionType().getNumResults(), std::nullopt),
                SmallVector<HandleFieldOriginSet>(
                    function.getFunctionType().getNumResults()),
                incomingHandles,
                incomingViews,
                incomingResourceCells,
                incomingPointerCells,
                {},
                {},
                {}};
      function.emitError(
          "resource provenance through recursion is unsupported");
      invalid = true;
      return {SmallVector<Fact>(),
              SmallVector<std::optional<int64_t>>(),
              SmallVector<std::optional<ObjectRefSet>>(),
              SmallVector<HandleFieldOriginSet>(),
              HandleMap(),
              ViewMap(),
              ResourceCellMap(),
              PointerCellMap(),
              {},
              {},
              {}};
    }
    std::string cacheKey;
    if (collectingGlobalSummaries) {
      cacheKey = analysisCacheKey(function, invocation);
      auto cached = analysisCache.find(cacheKey);
      if (cached != analysisCache.end() &&
          sameAnalysisInputs(cached->second, arguments, argumentIntegers,
                             argumentObjects, argumentHandleFields,
                             incomingHandles, incomingViews,
                             incomingResourceCells, incomingPointerCells)) {
        active.erase(function);
        if (cached->second.complete)
          return cached->second.result;
        return {cached->second.result.results,
                cached->second.result.integerResults,
                SmallVector<std::optional<ObjectRefSet>>(
                    function.getFunctionType().getNumResults(), std::nullopt),
                SmallVector<HandleFieldOriginSet>(
                    function.getFunctionType().getNumResults()),
                incomingHandles,
                incomingViews,
                incomingResourceCells,
                incomingPointerCells,
                {},
                {},
                {}};
      }
    }
    llvm::DenseMap<Value, Fact> facts;
    llvm::DenseMap<Value, int64_t> integerValues;
    llvm::DenseSet<Value> ambiguousIntegers;
    llvm::DenseMap<Value, ObjectRefSet> objectValues;
    llvm::DenseMap<Value, HandleFieldOriginSet> handleFieldValues;
    HandleMap handles = incomingHandles;
    ResourceCellMap initialResourceCells = incomingResourceCells;
    llvm::DenseSet<Value> eligibleCells;
    llvm::DenseSet<ObjectId> localObjects;
    function.walk([&](::vrc::udonhl::AllocaOp allocation) {
      if (isSingletonCell(allocation))
        eligibleCells.insert(allocation.getResult());
      auto bytes = constantAllocationBytes(allocation);
      if (bytes && (*bytes == 4 || *bytes == 8 ||
                    *bytes >= UDONCL_UDON_ABI_BUFFER_VIEW_BYTES)) {
        ObjectId object = localObject(invocation, allocation, *bytes);
        localObjects.insert(object);
        objectValues[allocation.getResult()] =
            ObjectRefSet::singleton({object, 0});
        if (*bytes == 8)
          handles.try_emplace(ObjectRef{object, 0}, Fact::nullHandle());
      }
    });
    function.walk([&](::vrc::udonhl::AddressOfOp address) {
      auto found = globalObjects.find(address.getGlobalAttr().getValue());
      if (found != globalObjects.end())
        objectValues[address.getResult()] =
            ObjectRefSet::singleton({found->getValue(), 0});
    });
    for (auto [index, argument] : llvm::enumerate(function.getArguments())) {
      Fact fact = index < arguments.size() ? arguments[index] : Fact::scalar();
      facts[argument] = fact;
      if (index < argumentIntegers.size() && argumentIntegers[index])
        integerValues[argument] = *argumentIntegers[index];
      if (index < argumentObjects.size() && argumentObjects[index])
        objectValues[argument] = *argumentObjects[index];
      if (index < argumentHandleFields.size() &&
          (!argumentHandleFields[index].refs.empty() ||
           argumentHandleFields[index].unknown))
        handleFieldValues[argument] = argumentHandleFields[index];
      recordCase(argumentCases, function, invocation, index, fact);
    }

    SmallVector<Fact> returns(function.getFunctionType().getNumResults());
    SmallVector<std::optional<int64_t>> integerReturns(
        function.getFunctionType().getNumResults(), std::nullopt);
    SmallVector<std::optional<ObjectRefSet>> objectReturns(
        function.getFunctionType().getNumResults(), std::nullopt);
    SmallVector<bool> objectReturnSeen(
        function.getFunctionType().getNumResults(), false);
    SmallVector<HandleFieldOriginSet> handleFieldReturns(
        function.getFunctionType().getNumResults());
    SmallVector<bool> handleFieldReturnSeen(
        function.getFunctionType().getNumResults(), false);
    SmallVector<bool> integerReturnSeen(
        function.getFunctionType().getNumResults(), false);
    SmallVector<Block *> worklist;
    llvm::DenseSet<Block *> queued;
    llvm::DenseSet<Block *> reached;
    std::map<ObjectRef, llvm::DenseSet<Block *>> handleReaders;
    llvm::DenseMap<Block *, CellMap> memoryInputs;
    llvm::DenseMap<std::pair<Block *, Block *>, CellMap> edgeMemory;
    llvm::DenseMap<Block *, PointerCellMap> pointerInputs;
    llvm::DenseMap<std::pair<Block *, Block *>, PointerCellMap> edgePointers;
    llvm::DenseMap<Block *, ResourceCellMap> resourceCellInputs;
    llvm::DenseMap<std::pair<Block *, Block *>, ResourceCellMap>
        edgeResourceCells;
    llvm::DenseMap<std::pair<Block *, Block *>,
                   SmallVector<std::optional<int64_t>>>
        edgeIntegerArguments;
    llvm::DenseMap<Block *, ViewMap> viewInputs;
    llvm::DenseMap<std::pair<Block *, Block *>, ViewMap> edgeViews;
    llvm::DenseMap<Block *, StatusConstraintMap> statusConstraintInputs;
    llvm::DenseMap<std::pair<Block *, Block *>, StatusConstraintMap>
        edgeStatusConstraints;
    llvm::DenseMap<Value, PendingMemoryEffect> pendingMemoryEffects;
    llvm::DenseSet<Operation *> trackedDomainTransparentOperations;
    llvm::DenseMap<Value, ComparisonRefinement> comparisonRefinements;
    PointerCellMap returnedPointerCells;
    ResourceCellMap returnedResourceCells;
    ViewMap returnedViews;
    llvm::DenseMap<Block *, ResourceCellExitCases> blockResourceCellExitCases;
    llvm::DenseMap<Block *, PointerExitCases> blockPointerExitCases;
    llvm::DenseMap<Block *, ViewExitCases> blockViewExitCases;
    bool sawReturn = false;
    llvm::DenseSet<Value> resourceCells;
    DominanceInfo dominance(function);
    auto nonNegativeUnitStrideLoop =
        [&](Value value, Operation *at) -> std::optional<int64_t> {
      auto argument = dyn_cast<BlockArgument>(value);
      if (!argument)
        return std::nullopt;
      Block *header = argument.getOwner();
      if (header->isEntryBlock() ||
          !dominance.dominates(header, at->getBlock()))
        return std::nullopt;
      std::optional<int64_t> start;
      bool sawBackedge = false;
      unsigned edges = 0;
      for (Block *predecessor : header->getPredecessors()) {
        ++edges;
        Value incoming =
            successorOperand(predecessor, header, argument.getArgNumber());
        if (!incoming)
          return std::nullopt;
        if (auto constant = incoming.getDefiningOp<arith::ConstantIntOp>()) {
          if (start || dominance.dominates(header, predecessor) ||
              constant.value() < 0)
            return std::nullopt;
          start = constant.value();
          continue;
        }
        if (sawBackedge || !dominance.dominates(header, predecessor))
          return std::nullopt;
        auto increment = incoming.getDefiningOp<arith::AddIOp>();
        if (!increment ||
            !arith::bitEnumContainsAll(increment.getOverflowFlags(),
                                       arith::IntegerOverflowFlags::nuw) ||
            !((increment.getLhs() == argument &&
               isConstantI32(increment.getRhs(), 1)) ||
              (increment.getRhs() == argument &&
               isConstantI32(increment.getLhs(), 1))))
          return std::nullopt;
        sawBackedge = true;
      }
      if (edges != 2 || !start || !sawBackedge)
        return std::nullopt;
      return start;
    };
    if (!function.empty()) {
      worklist.push_back(&function.front());
      queued.insert(&function.front());
      reached.insert(&function.front());
      pointerInputs[&function.front()] = incomingPointerCells;
      resourceCellInputs[&function.front()] = std::move(initialResourceCells);
      viewInputs[&function.front()] = incomingViews;
      statusConstraintInputs[&function.front()] = StatusConstraintMap();
      for (Value cell : eligibleCells)
        memoryInputs[&function.front()][cell] = {true, false, Fact()};
    }
    auto pendingEffectAtTerminator =
        [&](Value value, Block *block,
            Operation *terminator) -> PendingMemoryEffect * {
      auto pending = pendingMemoryEffects.find(value);
      if (pending == pendingMemoryEffects.end())
        return nullptr;
      Operation *next = nullptr;
      if (pending->second.call) {
        if (pending->second.call->getBlock() != block)
          return nullptr;
        next = pending->second.call->getNextNode();
      } else {
        auto argument = dyn_cast<BlockArgument>(value);
        if (!argument || argument.getOwner() != block)
          return nullptr;
        next = block->empty() ? nullptr : &block->front();
      }
      for (; next && next != terminator; next = next->getNextNode())
        if (!isa<arith::ConstantOp, arith::CmpIOp>(next) &&
            !trackedDomainTransparentOperations.contains(next))
          return nullptr;
      return next == terminator ? &pending->second : nullptr;
    };
    auto completeSuccessEffect =
        [&](Operation *call, const ResourceCellMap &failureResourceCells,
            const ResourceCellMap &successResourceCells,
            const PointerCellMap &pointerCells,
            const ViewMap &views) -> PendingMemoryEffect {
      PendingMemoryEffect effect;
      effect.call = call;
      addResourceCellExitCase(effect.resourceCellCases, 0,
                              successResourceCells);
      addResourceCellExitCase(effect.resourceCellCases, std::nullopt,
                              failureResourceCells, {0});
      addPointerExitCase(effect.pointerCases, 0, pointerCells,
                         zeroInitializedGlobalObjectIds);
      addPointerExitCase(effect.pointerCases, std::nullopt, pointerCells,
                         zeroInitializedGlobalObjectIds, {0});
      addViewExitCase(effect.viewCases, 0, views);
      addViewExitCase(effect.viewCases, std::nullopt, views, {0});
      return effect;
    };
    auto blockArgumentPendingEffect =
        [&](BlockArgument argument) -> std::optional<PendingMemoryEffect> {
      Block *destination = argument.getOwner();
      PendingMemoryEffect combined;
      bool sawReachableEdge = false;
      auto addPendingCases = [&](const PendingMemoryEffect &pending) {
        for (const auto &[status, cells] : pending.resourceCellCases.exact)
          addResourceCellExitCase(combined.resourceCellCases, status, cells);
        for (const auto &unknown : pending.resourceCellCases.unknown)
          addResourceCellExitCase(combined.resourceCellCases, std::nullopt,
                                  unknown.cells, unknown.excluded);
        for (const auto &[status, cells] : pending.pointerCases.exact)
          addPointerExitCase(combined.pointerCases, status, cells,
                             zeroInitializedGlobalObjectIds);
        for (const auto &unknown : pending.pointerCases.unknown)
          addPointerExitCase(combined.pointerCases, std::nullopt,
                             unknown.pointerCells,
                             zeroInitializedGlobalObjectIds, unknown.excluded);
        for (const auto &[status, views] : pending.viewCases.exact)
          addViewExitCase(combined.viewCases, status, views);
        for (const auto &unknown : pending.viewCases.unknown)
          addViewExitCase(combined.viewCases, std::nullopt, unknown.views,
                          unknown.excluded);
      };
      for (Block *predecessor : destination->getPredecessors()) {
        auto views = edgeViews.find({predecessor, destination});
        auto pointers = edgePointers.find({predecessor, destination});
        auto resourceCells = edgeResourceCells.find({predecessor, destination});
        if (views == edgeViews.end() && pointers == edgePointers.end() &&
            resourceCells == edgeResourceCells.end())
          continue;
        sawReachableEdge = true;
        Value incoming =
            successorOperand(predecessor, destination, argument.getArgNumber());
        if (isa<cf::BranchOp>(predecessor->getTerminator()))
          if (PendingMemoryEffect *pending = pendingEffectAtTerminator(
                  incoming, predecessor, predecessor->getTerminator())) {
            addPendingCases(*pending);
            continue;
          }
        std::optional<int64_t> status;
        auto integers = edgeIntegerArguments.find({predecessor, destination});
        if (integers != edgeIntegerArguments.end() &&
            argument.getArgNumber() < integers->second.size())
          status = integers->second[argument.getArgNumber()];
        StatusExclusions excluded;
        auto constraints =
            edgeStatusConstraints.find({predecessor, destination});
        if (constraints != edgeStatusConstraints.end()) {
          auto found = constraints->second.find(argument);
          if (found != constraints->second.end())
            excluded = found->second;
        }
        if (views != edgeViews.end())
          addViewExitCase(combined.viewCases, status, views->second, excluded);
        if (pointers != edgePointers.end())
          addPointerExitCase(combined.pointerCases, status, pointers->second,
                             zeroInitializedGlobalObjectIds, excluded);
        if (resourceCells != edgeResourceCells.end())
          addResourceCellExitCase(combined.resourceCellCases, status,
                                  resourceCells->second, excluded);
      }
      if (!sawReachableEdge)
        return std::nullopt;
      return combined;
    };
    llvm::DenseMap<Block *, unsigned> reversePostOrderIndex;
    if (!function.empty())
      for (auto [index, block] : llvm::enumerate(
               llvm::ReversePostOrderTraversal<Block *>(&function.front())))
        reversePostOrderIndex[block] = index;
    bool acyclic = llvm::all_of(reversePostOrderIndex, [&](const auto &entry) {
      return llvm::all_of(entry.first->getSuccessors(), [&](Block *successor) {
        return reversePostOrderIndex.lookup(successor) > entry.second;
      });
    });
    unsigned iterations = 0;
    size_t pushedByPreviousBlock = worklist.size();
    while (!worklist.empty() && !invalid) {
      if (++iterations > 10000) {
        function.emitError("device resource provenance did not converge");
        invalid = true;
        break;
      }
      // The worklist stays depth-first. In an acyclic CFG the blocks queued
      // by one visit are taken in reverse post-order; otherwise a join is
      // visited before its other forward predecessor, and a chain of
      // conditional regions revisits every later block once per pending
      // branch arm. Cyclic CFGs keep the established visit order because some
      // lifecycle checks still observe intermediate loop states.
      pushedByPreviousBlock = std::min(pushedByPreviousBlock, worklist.size());
      if (acyclic)
        std::stable_sort(worklist.begin() + pushedByPreviousBlock,
                         worklist.end(), [&](Block *lhs, Block *rhs) {
                           return reversePostOrderIndex.lookup(lhs) >
                                  reversePostOrderIndex.lookup(rhs);
                         });
      Block *block = worklist.pop_back_val();
      pushedByPreviousBlock = worklist.size();
      queued.erase(block);
      CellMap localMemory = memoryInputs[block];
      PointerCellMap localPointerCells = pointerInputs[block];
      ResourceCellMap localResourceCells = resourceCellInputs[block];
      ViewMap localViews = viewInputs[block];
      StatusConstraintMap localStatusConstraints =
          statusConstraintInputs[block];
      for (Operation &operation : *block) {
        for (auto [index, operand] : llvm::enumerate(operation.getOperands()))
          recordCase(operandCases, &operation, invocation, index,
                     valueFact(operand, facts));

        if (auto store = dyn_cast<::vrc::udonhl::StoreOp>(operation)) {
          Fact stored = valueFact(store.getValue(), facts);
          Value cell = store.getPointer();
          Fact pointer = valueFact(cell, facts);
          auto cellObjects = objectValues.find(cell);
          auto storedObjects = objectValues.find(store.getValue());
          auto exactCell = cellObjects == objectValues.end()
                               ? std::nullopt
                               : exactObject(cellObjects->second);
          auto storedOrigins = handleFieldValues.find(store.getValue());
          bool directViewField = false;
          if (exactCell) {
            directViewField =
                recordDirectViewField(*exactCell, store.getValue().getType(),
                                      storedOrigins == handleFieldValues.end()
                                          ? nullptr
                                          : &storedOrigins->second,
                                      localViews, handles);
          }
          bool existingObjectPointer = false;
          if (exactCell) {
            auto existing = localPointerCells.find(*exactCell);
            existingObjectPointer = existing != localPointerCells.end();
          }
          bool trackedObjectPointer =
              exactCell && (storedObjects != objectValues.end() ||
                            stored.zero || existingObjectPointer);
          if (trackedObjectPointer &&
              isHandleType(store.getValue().getType()) && !stored.resource) {
            PointerCellState incoming;
            incoming.hasPath = true;
            incoming.initializedOnAllPaths = true;
            if (storedObjects != objectValues.end())
              incoming.objects = storedObjects->second;
            else if (!stored.zero)
              incoming.objects.unknown = true;
            localPointerCells[*exactCell] = std::move(incoming);
          } else if (exactCell && isHandleType(store.getValue().getType()) &&
                     !stored.resource) {
            localPointerCells.erase(*exactCell);
          }
          if (exactCell && localResourceCells.count(*exactCell)) {
            Fact resourceValue = stored;
            if (!resourceValue.resource) {
              if (resourceValue.zero)
                resourceValue = Fact::nullHandle();
              else {
                resourceValue.initialized = true;
                resourceValue.resource = true;
                resourceValue.unknown = true;
              }
            }
            localResourceCells[*exactCell] = {true, true, resourceValue};
          }
          if (auto global = globalCell(cell)) {
            StringRef name = global.getSymName();
            bool classified = resourceGlobals.contains(name);
            if (!classified && stored.resource) {
              resourceGlobals.insert(name);
              classified = true;
            }
            if (classified && !stored.resource) {
              if (stored.zero)
                stored = Fact::nullHandle();
              else {
                stored.initialized = true;
                stored.resource = true;
                stored.unknown = true;
              }
            }
            globalWrites[name] = joinFacts(globalWrites[name], stored);
          } else if (eligibleCells.contains(cell)) {
            if (resourceCells.contains(cell) && !stored.resource) {
              if (stored.zero) {
                stored = Fact::nullHandle();
              } else {
                stored.initialized = true;
                stored.resource = true;
                stored.unknown = true;
              }
            }
            localMemory[cell] = {true, true, stored};
            if (stored.resource && resourceCells.insert(cell).second)
              for (Operation *user : cell.getUsers())
                if (reached.contains(user->getBlock()) &&
                    queued.insert(user->getBlock()).second)
                  worklist.push_back(user->getBlock());
          } else {
            if (stored.resource && !directViewField) {
              if (cell.getDefiningOp<::vrc::udonhl::AddressOfOp>())
                store.emitError("static device resource stored through an "
                                "unsupported or escaping global cell");
              else
                store.emitError("static device resource stored through a "
                                "non-singleton or escaping local cell");
              invalid = true;
            }
            if (pointer.resource) {
              store.emitError(
                  "static device resource used as a Host memory address");
              invalid = true;
            }
          }
        } else if (auto load = dyn_cast<::vrc::udonhl::LoadOp>(operation)) {
          Value cell = load.getPointer();
          auto cellObjects = objectValues.find(cell);
          auto exactCell = cellObjects == objectValues.end()
                               ? std::nullopt
                               : exactObject(cellObjects->second);
          if (auto global = globalCell(cell)) {
            updateFact(load.getResult(), globalFacts[global.getSymName()],
                       facts, worklist, queued, reached);
          } else if (exactCell && canHoldResourceCell(*exactCell) &&
                     localResourceCells.count(*exactCell)) {
            CellState state = localResourceCells[*exactCell];
            Fact loaded =
                state.initializedOnAllPaths ? state.fact : Fact::scalar();
            if (!state.initializedOnAllPaths && state.fact.resource) {
              loaded = state.fact;
              loaded.initialized = true;
              loaded.resource = true;
              loaded.unknown = true;
            }
            updateFact(load.getResult(), loaded, facts, worklist, queued,
                       reached);
          } else if (eligibleCells.contains(cell)) {
            CellState state = localMemory.lookup(cell);
            Fact loaded =
                state.initializedOnAllPaths ? state.fact : Fact::scalar();
            if (!state.initializedOnAllPaths && resourceCells.contains(cell)) {
              loaded = state.fact;
              loaded.initialized = true;
              loaded.resource = true;
              loaded.unknown = true;
            }
            updateFact(load.getResult(), loaded, facts, worklist, queued,
                       reached);
          } else if (valueFact(cell, facts).resource) {
            load.emitError(
                "static device resource used as a Host memory address");
            invalid = true;
          }
          if (exactCell && load.getType().isInteger(32)) {
            ObjectRef handle = *exactCell;
            std::optional<HandleFieldKind> kind;
            if (handle.offset >= 4) {
              ObjectRef generation{handle.base, handle.offset - 4};
              if (canHoldHandle(generation) && hasHandle(handles, generation)) {
                handle = generation;
                kind = HandleFieldKind::Generation;
              }
            }
            if (!kind && canHoldHandle(handle) && hasHandle(handles, handle))
              kind = HandleFieldKind::Slot;
            if (kind)
              handleFieldValues[load.getResult()] = {false, {{handle, *kind}}};
          }
          if (exactCell && isHandleType(load.getType()) &&
              !valueFact(load.getResult(), facts).resource) {
            auto pointerCell = localPointerCells.find(*exactCell);
            if (pointerCell != localPointerCells.end() &&
                pointerCell->second.hasPath) {
              if (!pointerCell->second.initializedOnAllPaths) {
                load.emitError("aggregate object pointer is not initialized "
                               "on every reaching path at object ")
                    << exactCell->base << " offset " << exactCell->offset
                    << " in @" << function.getSymName();
                invalid = true;
                continue;
              }
              objectValues[load.getResult()] = pointerCell->second.objects;
            }
          }
        } else if (auto memcpy = dyn_cast<::vrc::udonhl::MemcpyOp>(operation)) {
          copyViewSubobjects(memcpy.getDestination(), memcpy.getSource(),
                             memcpy.getByteSize(), objectValues, integerValues,
                             localViews);
        } else if (auto memmove =
                       dyn_cast<::vrc::udonhl::MemmoveOp>(operation)) {
          copyViewSubobjects(memmove.getDestination(), memmove.getSource(),
                             memmove.getByteSize(), objectValues, integerValues,
                             localViews);
        } else if (auto call = dyn_cast<func::CallOp>(operation)) {
          StringRef calleeName = call.getCallee();
          if (calleeName == "udonclResolvePhysicalBuffer") {
            analyzePhysicalBufferResolver(call, facts);
            for (Value result : call.getResults())
              updateFact(result, Fact::scalar(), facts, worklist, queued,
                         reached);
          } else if (calleeName == "udonclDeviceBufferCreate") {
            if (call.getNumOperands() != 4 || call.getNumResults() != 1 ||
                !call.getResult(0).getType().isInteger(32)) {
              call.emitError(
                  "udonclDeviceBufferCreate has an invalid signature");
              invalid = true;
              continue;
            }
            auto identity = allocationIdentity(invocation, call);
            auto object = objectValues.find(call.getOperand(3));
            auto exact = object == objectValues.end()
                             ? std::nullopt
                             : exactObject(object->second);
            if (!identity || !exact || !hasHandle(handles, *exact)) {
              call.emitError("udonclDeviceBufferCreate requires a discovered "
                             "allocation and an exact 8-byte handle object");
              invalid = true;
              continue;
            }
            Fact resource = Fact::identity(*identity);
            updateHandle(*exact, resource, handles, worklist, queued,
                         handleReaders);
            recordCase(operandCases, call, invocation, 3, resource);
            updateFact(call.getResult(0), Fact::scalar(), facts, worklist,
                       queued, reached);
          } else if (isAllocation(calleeName)) {
            if (call.getNumResults() != 1) {
              call.emitError(
                  "static device allocation must return exactly one handle");
              invalid = true;
              continue;
            }
            auto identity = allocationIdentity(invocation, call);
            if (identity)
              updateFact(call.getResult(0), Fact::identity(*identity), facts,
                         worklist, queued, reached);
          } else if (calleeName == "udonclDeviceBufferAdopt") {
            if (call.getNumOperands() != 4) {
              call.emitError("udonclDeviceBufferAdopt requires four operands");
              invalid = true;
              continue;
            }
            auto object = objectValues.find(call.getOperand(3));
            auto exact = object == objectValues.end()
                             ? std::nullopt
                             : exactObject(object->second);
            if (!exact || !hasHandle(handles, *exact)) {
              call.emitError("udonclDeviceBufferAdopt requires an exact "
                             "8-byte handle object");
              invalid = true;
              continue;
            }
            Fact resource = valueFact(call.getOperand(1), facts);
            if (!resource.resource) {
              call.emitError("udonclDeviceBufferAdopt requires a statically "
                             "known device resource");
              invalid = true;
              continue;
            }
            handleReaders[*exact].insert(block);
            Fact state =
                joinFacts(lookupHandle(handles, *exact), Fact::nullHandle());
            state = joinFacts(state, resource);
            updateHandle(*exact, state, handles, worklist, queued,
                         handleReaders);
            recordCase(operandCases, call, invocation, 3, state);
            for (Value result : call.getResults())
              updateFact(result, Fact::scalar(), facts, worklist, queued,
                         reached);
          } else if (calleeName == "udonclDeviceBufferGetRaw") {
            if (call.getNumOperands() != 2) {
              call.emitError("udonclDeviceBufferGetRaw requires two operands");
              invalid = true;
              continue;
            }
            auto object = objectValues.find(call.getOperand(0));
            auto exact = object == objectValues.end()
                             ? std::nullopt
                             : exactObject(object->second);
            if (!exact || !hasHandle(handles, *exact)) {
              call.emitError("udonclDeviceBufferGetRaw requires an exact "
                             "8-byte handle object");
              invalid = true;
              continue;
            }
            handleReaders[*exact].insert(block);
            Fact state = lookupHandle(handles, *exact);
            if (!state.initialized)
              state = Fact::nullHandle();
            recordCase(operandCases, call, invocation, 0, state);
            Value cell = call.getOperand(1);
            ResourceCellMap beforeResourceCells = localResourceCells;
            ResourceCellMap successfulResourceCells = localResourceCells;
            if (auto global = globalCell(cell)) {
              StringRef name = global.getSymName();
              resourceGlobals.insert(name);
              globalWrites[name] = joinFacts(globalWrites[name], state);
            } else if (auto object = objectValues.find(cell);
                       object != objectValues.end() &&
                       exactObject(object->second) &&
                       canHoldResourceCell(*exactObject(object->second))) {
              ObjectRef outputCell = *exactObject(object->second);
              CellState previous = successfulResourceCells[outputCell];
              Fact output = previous.initializedOnAllPaths
                                ? joinFacts(previous.fact, state)
                                : state;
              successfulResourceCells[outputCell] = {true, true, output};
              localResourceCells = successfulResourceCells;
            } else if (eligibleCells.contains(cell)) {
              CellState previous = localMemory.lookup(cell);
              Fact output = previous.initializedOnAllPaths
                                ? joinFacts(previous.fact, state)
                                : state;
              localMemory[cell] = {true, true, output};
              if (resourceCells.insert(cell).second)
                for (Operation *user : cell.getUsers()) {
                  Block *userBlock = user->getBlock();
                  if (reached.contains(userBlock) &&
                      queued.insert(userBlock).second)
                    worklist.push_back(userBlock);
                }
            } else {
              call.emitError("udonclDeviceBufferGetRaw requires an exact "
                             "4-byte output cell");
              invalid = true;
              continue;
            }
            for (Value result : call.getResults())
              updateFact(result, Fact::scalar(), facts, worklist, queued,
                         reached);
            if (call.getNumResults() == 1 &&
                successfulResourceCells != beforeResourceCells) {
              pendingMemoryEffects[call.getResult(0)] = completeSuccessEffect(
                  call, beforeResourceCells, successfulResourceCells,
                  localPointerCells, localViews);
            }
          } else if (calleeName == "udonclDeviceBufferDestroy") {
            if (call.getNumOperands() != 1) {
              call.emitError("udonclDeviceBufferDestroy requires one operand");
              invalid = true;
              continue;
            }
            auto object = objectValues.find(call.getOperand(0));
            auto exact = object == objectValues.end()
                             ? std::nullopt
                             : exactObject(object->second);
            if (!exact || !hasHandle(handles, *exact)) {
              call.emitError("udonclDeviceBufferDestroy requires an exact "
                             "8-byte handle object: ")
                  << call.getOperand(0);
              invalid = true;
              continue;
            }
            handleReaders[*exact].insert(block);
            Fact state = lookupHandle(handles, *exact);
            if (!state.initialized)
              state = Fact::nullHandle();
            recordCase(operandCases, call, invocation, 0, state);
            updateHandle(*exact, Fact::nullHandle(), handles, worklist, queued,
                         handleReaders);
            for (Value result : call.getResults())
              updateFact(result, Fact::scalar(), facts, worklist, queued,
                         reached);
          } else if (calleeName == "udonclDeviceBufferView") {
            if (call.getNumOperands() != 5) {
              call.emitError("udonclDeviceBufferView requires five operands");
              invalid = true;
              continue;
            }
            auto handle = objectValues.find(call.getOperand(0));
            auto output = objectValues.find(call.getOperand(4));
            auto exactHandle = handle == objectValues.end()
                                   ? std::nullopt
                                   : exactObject(handle->second);
            auto exactOutput = output == objectValues.end()
                                   ? std::nullopt
                                   : exactObject(output->second);
            if (!exactHandle || !hasHandle(handles, *exactHandle)) {
              call.emitError("udonclDeviceBufferView requires an exact "
                             "8-byte handle object");
              invalid = true;
              continue;
            }
            if (!exactOutput || !canHoldView(*exactOutput)) {
              call.emitError("udonclDeviceBufferView requires an exact ")
                  << UDONCL_UDON_ABI_BUFFER_VIEW_BYTES
                  << "-byte view subobject";
              invalid = true;
              continue;
            }
            handleReaders[*exactHandle].insert(block);
            Fact resource = lookupHandle(handles, *exactHandle);
            ViewState previous = localViews[*exactOutput];
            ViewState next;
            next.hasPath = true;
            next.initialized = true;
            next.mayBeInvalid = true;
            next.resource = previous.initialized
                                ? joinFacts(previous.resource, resource)
                                : resource;
            localViews[*exactOutput] = next;
            recordGlobalViewWrite(*exactOutput, next);
            recordCase(operandCases, call, invocation, 0, resource);
            recordCase(operandCases, call, invocation, 4, next.resource);
            for (Value result : call.getResults())
              updateFact(result, Fact::scalar(), facts, worklist, queued,
                         reached);
          } else if (calleeName == "udononnxCopyBufferView") {
            if (call.getNumOperands() != 2) {
              call.emitError("udononnxCopyBufferView requires two operands");
              invalid = true;
              continue;
            }
            auto destination = objectValues.find(call.getOperand(0));
            auto source = objectValues.find(call.getOperand(1));
            auto exactDestination = destination == objectValues.end()
                                        ? std::nullopt
                                        : exactObject(destination->second);
            auto exactSource = source == objectValues.end()
                                   ? std::nullopt
                                   : exactObject(source->second);
            if (!exactSource || !canHoldView(*exactSource) ||
                !localViews[*exactSource].initialized ||
                !localViews[*exactSource].resource.resource ||
                localViews[*exactSource].resource.unknown) {
              if (collectingGlobalSummaries)
                continue;
              call.emitError("udononnxCopyBufferView requires an initialized "
                             "source view subobject in @")
                  << function.getSymName() << ": " << call.getOperand(1);
              invalid = true;
              continue;
            }
            if (!exactDestination || !canHoldView(*exactDestination)) {
              if (collectingGlobalSummaries)
                continue;
              call.emitError("udononnxCopyBufferView requires an exact "
                             "destination view subobject in @")
                  << function.getSymName() << " from root @"
                  << invocation.root.getValue() << ": " << call.getOperand(0);
              invalid = true;
              continue;
            }
            localViews[*exactDestination] = localViews[*exactSource];
            recordGlobalViewWrite(*exactDestination,
                                  localViews[*exactDestination]);
            recordCase(operandCases, call, invocation, 0,
                       localViews[*exactDestination].resource);
            recordCase(operandCases, call, invocation, 1,
                       localViews[*exactSource].resource);
            for (Value result : call.getResults())
              updateFact(result, Fact::scalar(), facts, worklist, queued,
                         reached);
          } else if (calleeName == "udonclBufferViewResolve") {
            if (call.getNumOperands() != 4) {
              call.emitError("udonclBufferViewResolve requires four operands");
              invalid = true;
              continue;
            }
            auto view = objectValues.find(call.getOperand(0));
            Fact resource;
            bool hasResource = false;
            bool validView = view != objectValues.end() &&
                             !view->second.unknown &&
                             !view->second.refs.empty();
            if (validView)
              for (ObjectRef candidate : view->second.refs) {
                auto state = localViews.find(candidate);
                if (!canHoldView(candidate) || state == localViews.end() ||
                    !state->second.initialized ||
                    !state->second.resource.resource ||
                    state->second.resource.unknown) {
                  validView = false;
                  break;
                }
                resource = hasResource
                               ? joinFacts(resource, state->second.resource)
                               : state->second.resource;
                hasResource = true;
              }
            if (!validView || !hasResource) {
              if (collectingGlobalSummaries)
                continue;
              call.emitError("udonclBufferViewResolve requires an initialized "
                             "view subobject in @")
                  << function.getSymName() << " from root @"
                  << invocation.root.getValue() << ": " << call.getOperand(0);
              invalid = true;
              continue;
            }
            recordCase(operandCases, call, invocation, 0, resource);
            // A successful view resolution has found a live Device-buffer
            // table entry whose cl_mem is non-null.  Keep the input operand's
            // conservative nullability for diagnostics, but refine the
            // out-parameter on the status==success edge.  Interprocedural
            // memory-effect summaries preserve this correlation through
            // ordinary wrapper functions.
            Fact successfulResource = resource;
            successfulResource.mayBeNull = false;
            Value cell = call.getOperand(1);
            ResourceCellMap beforeResourceCells = localResourceCells;
            ResourceCellMap successfulResourceCells = localResourceCells;
            if (auto global = globalCell(cell)) {
              StringRef name = global.getSymName();
              resourceGlobals.insert(name);
              globalWrites[name] = joinFacts(globalWrites[name], resource);
            } else if (auto object = objectValues.find(cell);
                       object != objectValues.end() &&
                       exactObject(object->second) &&
                       canHoldResourceCell(*exactObject(object->second))) {
              ObjectRef outputCell = *exactObject(object->second);
              CellState previous = successfulResourceCells[outputCell];
              Fact output = previous.initializedOnAllPaths
                                ? joinFacts(previous.fact, successfulResource)
                                : successfulResource;
              successfulResourceCells[outputCell] = {true, true, output};
              localResourceCells = successfulResourceCells;
            } else if (eligibleCells.contains(cell)) {
              CellState previous = localMemory.lookup(cell);
              Fact output = previous.initializedOnAllPaths
                                ? joinFacts(previous.fact, successfulResource)
                                : successfulResource;
              localMemory[cell] = {true, true, output};
              resourceCells.insert(cell);
            } else {
              call.emitError("udonclBufferViewResolve requires an exact "
                             "4-byte cl_mem output cell");
              invalid = true;
              continue;
            }
            for (Value result : call.getResults())
              updateFact(result, Fact::scalar(), facts, worklist, queued,
                         reached);
            if (call.getNumResults() == 1 &&
                successfulResourceCells != beforeResourceCells) {
              pendingMemoryEffects[call.getResult(0)] = completeSuccessEffect(
                  call, beforeResourceCells, successfulResourceCells,
                  localPointerCells, localViews);
            }
          } else if (auto found = functions.find(calleeName);
                     found != functions.end() &&
                     !found->getValue().isExternal()) {
            SmallVector<Fact> actuals;
            SmallVector<std::optional<int64_t>> actualIntegers;
            SmallVector<std::optional<ObjectRefSet>> actualObjects;
            SmallVector<HandleFieldOriginSet> actualHandleFields;
            for (Value operand : call.getOperands()) {
              actuals.push_back(valueFact(operand, facts));
              auto integer = integerValues.find(operand);
              actualIntegers.push_back(
                  integer == integerValues.end()
                      ? std::nullopt
                      : std::optional<int64_t>(integer->second));
              auto object = objectValues.find(operand);
              actualObjects.push_back(
                  object == objectValues.end()
                      ? std::nullopt
                      : std::optional<ObjectRefSet>(object->second));
              auto handleField = handleFieldValues.find(operand);
              actualHandleFields.push_back(handleField ==
                                                   handleFieldValues.end()
                                               ? HandleFieldOriginSet()
                                               : handleField->second);
            }
            Invocation nested = calleeInvocation(invocation, function, call);
            AnalysisResult nestedResult =
                analyze(found->getValue(), nested, actuals, actualIntegers,
                        actualObjects, actualHandleFields, handles, localViews,
                        localResourceCells, localPointerCells);
            auto resourceCasesMatchFinal = [&] {
              return llvm::all_of(nestedResult.resourceCellExitCases.exact,
                                  [&](const auto &entry) {
                                    return entry.second ==
                                           nestedResult.resourceCells;
                                  }) &&
                     llvm::all_of(nestedResult.resourceCellExitCases.unknown,
                                  [&](const auto &entry) {
                                    return entry.cells ==
                                           nestedResult.resourceCells;
                                  });
            };
            auto pointerCasesMatchFinal = [&] {
              return llvm::all_of(nestedResult.pointerExitCases.exact,
                                  [&](const auto &entry) {
                                    return entry.second ==
                                           nestedResult.pointerCells;
                                  }) &&
                     llvm::all_of(nestedResult.pointerExitCases.unknown,
                                  [&](const auto &entry) {
                                    return entry.pointerCells ==
                                           nestedResult.pointerCells;
                                  });
            };
            auto viewCasesMatchFinal = [&] {
              return llvm::all_of(nestedResult.viewExitCases.exact,
                                  [&](const auto &entry) {
                                    return entry.second == nestedResult.views;
                                  }) &&
                     llvm::all_of(nestedResult.viewExitCases.unknown,
                                  [&](const auto &entry) {
                                    return entry.views == nestedResult.views;
                                  });
            };
            bool hasConditionalMemoryEffect = !resourceCasesMatchFinal() ||
                                              !pointerCasesMatchFinal() ||
                                              !viewCasesMatchFinal();
            if (!hasConditionalMemoryEffect &&
                nestedResult.handles == handles &&
                nestedResult.views == localViews &&
                nestedResult.resourceCells == localResourceCells &&
                nestedResult.pointerCells == localPointerCells)
              trackedDomainTransparentOperations.insert(call);
            handles = std::move(nestedResult.handles);
            localViews = std::move(nestedResult.views);
            localResourceCells = std::move(nestedResult.resourceCells);
            localPointerCells = std::move(nestedResult.pointerCells);
            if (call.getNumResults() == 1 && hasConditionalMemoryEffect)
              pendingMemoryEffects[call.getResult(0)] = {
                  call, std::move(nestedResult.resourceCellExitCases),
                  std::move(nestedResult.pointerExitCases),
                  std::move(nestedResult.viewExitCases)};
            for (auto [index, result] : llvm::enumerate(call.getResults()))
              updateFact(result,
                         index < nestedResult.results.size()
                             ? nestedResult.results[index]
                             : Fact::scalar(),
                         facts, worklist, queued, reached);
            for (auto [index, result] : llvm::enumerate(call.getResults()))
              mergeExactInteger(result,
                                index < nestedResult.integerResults.size()
                                    ? nestedResult.integerResults[index]
                                    : std::nullopt,
                                integerValues, ambiguousIntegers);
            for (auto [index, result] : llvm::enumerate(call.getResults()))
              if (index < nestedResult.objectResults.size() &&
                  nestedResult.objectResults[index])
                objectValues[result] = *nestedResult.objectResults[index];
            for (auto [index, result] : llvm::enumerate(call.getResults()))
              if (index < nestedResult.handleFieldResults.size() &&
                  (!nestedResult.handleFieldResults[index].refs.empty() ||
                   nestedResult.handleFieldResults[index].unknown))
                handleFieldValues[result] =
                    nestedResult.handleFieldResults[index];
          } else {
            bool hasResource =
                llvm::any_of(call.getOperands(), [&](Value value) {
                  return valueFact(value, facts).resource;
                });
            bool hasUnknown =
                llvm::any_of(call.getOperands(), [&](Value value) {
                  Fact fact = valueFact(value, facts);
                  return fact.resource && fact.unknown;
                });
            if (hasUnknown) {
              call.emitError(
                  "resource use has unresolved non-handle provenance in call ")
                  << calleeName << " from @" << function.getSymName();
              invalid = true;
            }
            if (hasResource && !isKnownOpenCLConsumer(calleeName) &&
                !hasResourceOperandContract(call, facts)) {
              call.emitError(
                  "static device resource escapes to unknown external call ")
                  << calleeName << " from @" << function.getSymName();
              invalid = true;
            }
            for (Value result : call.getResults())
              updateFact(result, Fact::scalar(), facts, worklist, queued,
                         reached);
          }
        } else if (auto constant = dyn_cast<arith::ConstantIntOp>(operation)) {
          integerValues[constant.getResult()] = constant.value();
          updateFact(constant.getResult(),
                     constant.value() == 0 ? Fact::zeroScalar()
                                           : Fact::scalar(),
                     facts, worklist, queued, reached);
        } else if (auto intToPtr =
                       dyn_cast<::vrc::udonhl::IntToPtrOp>(operation)) {
          Fact address = valueFact(intToPtr.getAddress(), facts);
          updateFact(intToPtr.getResult(),
                     address.resource
                         ? address
                         : (address.zero ? Fact::zeroScalar() : Fact::scalar()),
                     facts, worklist, queued, reached);
        } else if (auto pointerToInt =
                       dyn_cast<::vrc::udonhl::PtrToIntOp>(operation)) {
          Fact pointer = valueFact(pointerToInt.getPointer(), facts);
          updateFact(pointerToInt.getResult(),
                     pointer.resource
                         ? pointer
                         : (pointer.zero ? Fact::zeroScalar() : Fact::scalar()),
                     facts, worklist, queued, reached);
        } else if (auto pointerAdd =
                       dyn_cast<::vrc::udonhl::PtrAddOp>(operation)) {
          // A loop value may widen after an earlier exact visit. Never retain
          // the stale exact address when recomputing its pointer addition.
          objectValues.erase(pointerAdd.getResult());
          auto base = objectValues.find(pointerAdd.getPointer());
          auto offset = integerValues.find(pointerAdd.getByteOffset());
          if (base != objectValues.end() && offset != integerValues.end()) {
            ObjectRefSet shifted;
            shifted.unknown = base->second.unknown;
            for (ObjectRef object : base->second.refs) {
              int64_t combined =
                  static_cast<int64_t>(object.offset) + offset->second;
              if (combined < std::numeric_limits<int32_t>::min() ||
                  combined > std::numeric_limits<int32_t>::max()) {
                shifted.unknown = true;
                continue;
              }
              shifted.refs.push_back(
                  {object.base, static_cast<int32_t>(combined)});
            }
            objectValues[pointerAdd.getResult()] = std::move(shifted);
          } else if (base != objectValues.end()) {
            Value induction = pointerAdd.getByteOffset();
            int64_t stride = 1;
            bool explicitStride = false;
            if (auto multiply = induction.getDefiningOp<arith::MulIOp>()) {
              auto left =
                  multiply.getLhs().getDefiningOp<arith::ConstantIntOp>();
              auto right =
                  multiply.getRhs().getDefiningOp<arith::ConstantIntOp>();
              if (left && left.value() > 0) {
                stride = left.value();
                induction = multiply.getRhs();
                explicitStride = true;
              } else if (right && right.value() > 0) {
                stride = right.value();
                induction = multiply.getLhs();
                explicitStride = true;
              } else {
                induction = Value();
              }
            }
            auto start = induction
                             ? nonNegativeUnitStrideLoop(induction, pointerAdd)
                             : std::nullopt;
            // An index that is not a recognized loop induction (for example
            // a function argument whose callers pass different values) still
            // selects one element of an explicitly strided array in every
            // defined C execution. Without a known lower bound, enumerate the
            // in-bounds elements on both sides of the base subobject.
            bool anyElement = !start && induction && explicitStride;
            if ((start || anyElement) && stride > 0) {
              constexpr size_t maxAlternatives = 64;
              ObjectRefSet shifted;
              shifted.unknown = base->second.unknown;
              for (ObjectRef object : base->second.refs) {
                auto size = objectSizes.find(object.base);
                if (size == objectSizes.end() ||
                    (!start && object.offset < 0) ||
                    (start &&
                     *start > std::numeric_limits<int64_t>::max() / stride)) {
                  shifted.unknown = true;
                  continue;
                }
                int64_t first = start ? *start * stride
                                      : -(object.offset / stride) * stride;
                int64_t extent = size->second;
                // A declared buffer-view layout with the same element stride
                // names the element array held by the storage. Any tail past
                // its last element is reserved storage, not an element that
                // an index can select.
                if (auto layout = restrictedBufferViewLayouts.find(object.base);
                    layout != restrictedBufferViewLayouts.end() &&
                    layout->second.stride == stride) {
                  int64_t elements =
                      static_cast<int64_t>(layout->second.count) * stride;
                  if (object.offset < elements)
                    extent = std::min(extent, elements);
                }
                // C executions that dereference this pointer must remain
                // inside the statically identified object. Enumerate every
                // constant-stride subobject that such a defined execution can
                // select; later consumers validate every candidate.
                for (int64_t relative = first;
                     relative < extent - object.offset; relative += stride) {
                  if (shifted.refs.size() >= maxAlternatives ||
                      relative >
                          std::numeric_limits<int32_t>::max() - object.offset) {
                    shifted.unknown = true;
                    break;
                  }
                  shifted.refs.push_back(
                      {object.base,
                       static_cast<int32_t>(object.offset + relative)});
                  if (relative > std::numeric_limits<int64_t>::max() - stride)
                    break;
                }
              }
              if (!shifted.refs.empty())
                objectValues[pointerAdd.getResult()] = std::move(shifted);
            }
          }
          updateFact(pointerAdd.getResult(), Fact::scalar(), facts, worklist,
                     queued, reached);
        } else if (auto add = dyn_cast<arith::AddIOp>(operation)) {
          if (valueFact(add.getLhs(), facts).resource ||
              valueFact(add.getRhs(), facts).resource) {
            add.emitError(
                "unsupported operation consumes a static device resource: ")
                << add.getOperationName() << " in @" << function.getSymName();
            invalid = true;
          }
          auto left = integerValues.find(add.getLhs());
          auto right = integerValues.find(add.getRhs());
          std::optional<int64_t> folded;
          if (left != integerValues.end() && right != integerValues.end() &&
              !ambiguousIntegers.contains(add.getLhs()) &&
              !ambiguousIntegers.contains(add.getRhs()))
            folded = foldExactInteger(
                add.getType(), left->second, right->second,
                [](const llvm::APInt &lhs, const llvm::APInt &rhs) {
                  return lhs + rhs;
                });
          mergeExactInteger(add.getResult(), folded, integerValues,
                            ambiguousIntegers);
          updateFact(add.getResult(), Fact::scalar(), facts, worklist, queued,
                     reached);
        } else if (auto subtract = dyn_cast<arith::SubIOp>(operation)) {
          if (valueFact(subtract.getLhs(), facts).resource ||
              valueFact(subtract.getRhs(), facts).resource) {
            subtract.emitError(
                "unsupported operation consumes a static device resource: ")
                << subtract.getOperationName() << " in @"
                << function.getSymName();
            invalid = true;
          }
          auto left = integerValues.find(subtract.getLhs());
          auto right = integerValues.find(subtract.getRhs());
          std::optional<int64_t> folded;
          if (left != integerValues.end() && right != integerValues.end() &&
              !ambiguousIntegers.contains(subtract.getLhs()) &&
              !ambiguousIntegers.contains(subtract.getRhs()))
            folded = foldExactInteger(
                subtract.getType(), left->second, right->second,
                [](const llvm::APInt &lhs, const llvm::APInt &rhs) {
                  return lhs - rhs;
                });
          mergeExactInteger(subtract.getResult(), folded, integerValues,
                            ambiguousIntegers);
          updateFact(subtract.getResult(), Fact::scalar(), facts, worklist,
                     queued, reached);
        } else if (auto multiply = dyn_cast<arith::MulIOp>(operation)) {
          if (valueFact(multiply.getLhs(), facts).resource ||
              valueFact(multiply.getRhs(), facts).resource) {
            multiply.emitError(
                "unsupported operation consumes a static device resource: ")
                << multiply.getOperationName() << " in @"
                << function.getSymName();
            invalid = true;
          }
          auto left = integerValues.find(multiply.getLhs());
          auto right = integerValues.find(multiply.getRhs());
          std::optional<int64_t> folded;
          if (left != integerValues.end() && right != integerValues.end() &&
              !ambiguousIntegers.contains(multiply.getLhs()) &&
              !ambiguousIntegers.contains(multiply.getRhs()))
            folded = foldExactInteger(
                multiply.getType(), left->second, right->second,
                [](const llvm::APInt &lhs, const llvm::APInt &rhs) {
                  return lhs * rhs;
                });
          mergeExactInteger(multiply.getResult(), folded, integerValues,
                            ambiguousIntegers);
          updateFact(multiply.getResult(), Fact::scalar(), facts, worklist,
                     queued, reached);
        } else if (auto compare = dyn_cast<arith::CmpIOp>(operation)) {
          Fact leftFact = valueFact(compare.getLhs(), facts);
          Fact rightFact = valueFact(compare.getRhs(), facts);
          bool comparesResource = leftFact.resource || rightFact.resource;
          bool equality = compare.getPredicate() == arith::CmpIPredicate::eq ||
                          compare.getPredicate() == arith::CmpIPredicate::ne;
          auto isFiniteResource = [](const Fact &fact) {
            return fact.resource && fact.initialized && !fact.unknown &&
                   !fact.ids.empty();
          };
          bool comparesResourceWithNull =
              equality && ((leftFact.resource && rightFact.zero) ||
                           (rightFact.resource && leftFact.zero));
          if (comparesResource && !comparesResourceWithNull) {
            compare.emitError(
                "unsupported operation consumes a static device resource: ")
                << compare.getOperationName() << " in @"
                << function.getSymName();
            invalid = true;
          }
          const Fact &resourceFact = leftFact.resource ? leftFact : rightFact;
          std::optional<int64_t> foldedComparison;
          if (comparesResourceWithNull && isFiniteResource(resourceFact) &&
              !resourceFact.mayBeNull)
            foldedComparison =
                compare.getPredicate() == arith::CmpIPredicate::ne ? 1 : 0;
          auto leftInteger = integerValues.find(compare.getLhs());
          auto rightInteger = integerValues.find(compare.getRhs());
          if (!comparesResource && leftInteger != integerValues.end() &&
              rightInteger != integerValues.end() &&
              !ambiguousIntegers.contains(compare.getLhs()) &&
              !ambiguousIntegers.contains(compare.getRhs()))
            foldedComparison = foldExactIntegerComparison(
                compare.getLhs().getType(), leftInteger->second,
                rightInteger->second, compare.getPredicate());
          mergeExactInteger(compare.getResult(), foldedComparison,
                            integerValues, ambiguousIntegers);
          if (compare.getPredicate() == arith::CmpIPredicate::eq ||
              compare.getPredicate() == arith::CmpIPredicate::ne) {
            for (Value operand : compare.getOperands()) {
              auto argument = dyn_cast<BlockArgument>(operand);
              if (!argument)
                continue;
              if (auto pending = blockArgumentPendingEffect(argument))
                pendingMemoryEffects[argument] = std::move(*pending);
            }
            Value subject;
            std::optional<int64_t> constant;
            if (pendingMemoryEffects.contains(compare.getLhs())) {
              subject = compare.getLhs();
              auto found = integerValues.find(compare.getRhs());
              if (found != integerValues.end())
                constant = found->second;
            } else if (pendingMemoryEffects.contains(compare.getRhs())) {
              subject = compare.getRhs();
              auto found = integerValues.find(compare.getLhs());
              if (found != integerValues.end())
                constant = found->second;
            }
            if (subject && constant) {
              comparisonRefinements[compare.getResult()] = {
                  subject, *constant,
                  compare.getPredicate() == arith::CmpIPredicate::eq};
              pendingMemoryEffects[compare.getResult()] =
                  transformPendingComparison(
                      compare, pendingMemoryEffects[subject], *constant,
                      compare.getPredicate() == arith::CmpIPredicate::eq,
                      zeroInitializedGlobalObjectIds);
            } else if (!foldedComparison) {
              // A value without its own memory effect still carries the
              // branch outcome as a status exclusion. A later join may pass
              // it alongside an effectful status, and the exclusion is what
              // keeps its edge out of the effectful status's success case.
              auto exactConstant = [&](Value value) -> std::optional<int64_t> {
                auto found = integerValues.find(value);
                if (found == integerValues.end() ||
                    ambiguousIntegers.contains(value))
                  return std::nullopt;
                return found->second;
              };
              std::optional<int64_t> left = exactConstant(compare.getLhs());
              std::optional<int64_t> right = exactConstant(compare.getRhs());
              if (left.has_value() != right.has_value())
                comparisonRefinements[compare.getResult()] = {
                    left ? compare.getRhs() : compare.getLhs(),
                    left ? *left : *right,
                    compare.getPredicate() == arith::CmpIPredicate::eq};
            }
          }
          updateFact(compare.getResult(), Fact::scalar(), facts, worklist,
                     queued, reached);
        } else if (auto extend = dyn_cast<arith::ExtUIOp>(operation)) {
          auto integer = integerValues.find(extend.getIn());
          mergeExactInteger(extend.getOut(),
                            integer != integerValues.end() &&
                                    !ambiguousIntegers.contains(extend.getIn())
                                ? std::optional<int64_t>(integer->second)
                                : std::nullopt,
                            integerValues, ambiguousIntegers);
          auto pending = pendingMemoryEffects.find(extend.getIn());
          if (pending != pendingMemoryEffects.end()) {
            PendingMemoryEffect forwarded = pending->second;
            forwarded.call = extend;
            pendingMemoryEffects[extend.getOut()] = std::move(forwarded);
          }
          updateFact(extend.getOut(), Fact::scalar(), facts, worklist, queued,
                     reached);
        } else if (auto select = dyn_cast<arith::SelectOp>(operation)) {
          if (isHandleType(select.getType()))
            objectValues[select.getResult()] = joinObjectRefs(
                objectAlternative(select.getTrueValue(), facts, objectValues),
                objectAlternative(select.getFalseValue(), facts, objectValues));
          auto trueInteger = integerValues.find(select.getTrueValue());
          auto falseInteger = integerValues.find(select.getFalseValue());
          std::optional<int64_t> selectedInteger;
          if (trueInteger != integerValues.end() &&
              falseInteger != integerValues.end() &&
              !ambiguousIntegers.contains(select.getTrueValue()) &&
              !ambiguousIntegers.contains(select.getFalseValue()) &&
              trueInteger->second == falseInteger->second)
            selectedInteger = trueInteger->second;
          mergeExactInteger(select.getResult(), selectedInteger, integerValues,
                            ambiguousIntegers);
          auto trueHandleField = handleFieldValues.find(select.getTrueValue());
          auto falseHandleField =
              handleFieldValues.find(select.getFalseValue());
          if (trueHandleField != handleFieldValues.end() ||
              falseHandleField != handleFieldValues.end()) {
            HandleFieldOriginSet selected;
            if (trueHandleField != handleFieldValues.end())
              selected = trueHandleField->second;
            else
              selected.unknown = true;
            if (falseHandleField != handleFieldValues.end())
              selected = joinHandleFieldOrigins(std::move(selected),
                                                falseHandleField->second);
            else
              selected.unknown = true;
            handleFieldValues[select.getResult()] = std::move(selected);
          }
          updateFact(select.getResult(),
                     joinFacts(valueFact(select.getTrueValue(), facts),
                               valueFact(select.getFalseValue(), facts)),
                     facts, worklist, queued, reached);
        } else if (auto branch = dyn_cast<cf::BranchOp>(operation)) {
          if (branch.getDest()->getSinglePredecessor() == block) {
            for (auto [operand, argument] :
                 llvm::zip_equal(branch.getDestOperands(),
                                 branch.getDest()->getArguments())) {
              PendingMemoryEffect *pending =
                  pendingEffectAtTerminator(operand, block, &operation);
              if (!pending)
                continue;
              PendingMemoryEffect forwarded = *pending;
              // A null call denotes an effect available at the destination
              // block entry. The conditional-branch refinement below accepts
              // it only when no intervening operation can change that state.
              forwarded.call = nullptr;
              pendingMemoryEffects[argument] = std::move(forwarded);
            }
          }
          mergeSuccessor(branch.getDest(), branch.getDestOperands(), facts,
                         worklist, queued, reached, block, localMemory,
                         memoryInputs, edgeMemory, eligibleCells, objectValues,
                         integerValues, ambiguousIntegers, handleFieldValues,
                         localViews, viewInputs, edgeViews, localPointerCells,
                         pointerInputs, edgePointers, localResourceCells,
                         resourceCellInputs, edgeResourceCells,
                         localStatusConstraints, statusConstraintInputs,
                         edgeStatusConstraints, edgeIntegerArguments);
        } else if (auto branch = dyn_cast<cf::CondBranchOp>(operation)) {
          PointerCellMap truePointers = localPointerCells;
          PointerCellMap falsePointers = localPointerCells;
          ResourceCellMap trueResourceCells = localResourceCells;
          ResourceCellMap falseResourceCells = localResourceCells;
          ViewMap trueViews = localViews;
          ViewMap falseViews = localViews;
          bool trueFeasible = true;
          bool falseFeasible = true;
          StatusConstraintMap trueStatusConstraints = localStatusConstraints;
          StatusConstraintMap falseStatusConstraints = localStatusConstraints;
          auto refinement = comparisonRefinements.find(branch.getCondition());
          if (refinement != comparisonRefinements.end()) {
            StatusConstraintMap &notEqualConstraints =
                refinement->second.equalOnTrue ? falseStatusConstraints
                                               : trueStatusConstraints;
            notEqualConstraints[refinement->second.subject].insert(
                refinement->second.constant);
            auto effect = pendingMemoryEffects.find(refinement->second.subject);
            Operation *comparison = branch.getCondition().getDefiningOp();
            bool direct = effect != pendingMemoryEffects.end() && comparison &&
                          comparison->getBlock() == block;
            if (direct && effect->second.call) {
              direct = effect->second.call->getBlock() == block;
              for (Operation *between = effect->second.call->getNextNode();
                   between && between != &operation;
                   between = between->getNextNode())
                if (between != comparison) {
                  direct = false;
                  break;
                }
            } else if (direct) {
              auto argument =
                  dyn_cast<BlockArgument>(refinement->second.subject);
              direct = argument && argument.getOwner() == block;
              if (direct)
                for (Operation &between : *block) {
                  if (&between == &operation)
                    break;
                  if (&between != comparison &&
                      !isa<arith::ConstantOp>(between)) {
                    direct = false;
                    break;
                  }
                }
            }
            if (direct) {
              auto hasCase = [&](bool equal) {
                return hasMatchingExitCase(effect->second.pointerCases,
                                           refinement->second.constant,
                                           equal) ||
                       hasMatchingExitCase(effect->second.viewCases,
                                           refinement->second.constant,
                                           equal) ||
                       hasMatchingExitCase(effect->second.resourceCellCases,
                                           refinement->second.constant, equal);
              };
              trueFeasible = hasCase(refinement->second.equalOnTrue);
              falseFeasible = hasCase(!refinement->second.equalOnTrue);
              truePointers = selectPointerExitCases(
                  effect->second.pointerCases, refinement->second.constant,
                  refinement->second.equalOnTrue, localPointerCells,
                  zeroInitializedGlobalObjectIds);
              falsePointers = selectPointerExitCases(
                  effect->second.pointerCases, refinement->second.constant,
                  !refinement->second.equalOnTrue, localPointerCells,
                  zeroInitializedGlobalObjectIds);
              trueViews = selectViewExitCases(
                  effect->second.viewCases, refinement->second.constant,
                  refinement->second.equalOnTrue, localViews);
              falseViews = selectViewExitCases(
                  effect->second.viewCases, refinement->second.constant,
                  !refinement->second.equalOnTrue, localViews);
              trueResourceCells = selectResourceCellExitCases(
                  effect->second.resourceCellCases, refinement->second.constant,
                  refinement->second.equalOnTrue, localResourceCells);
              falseResourceCells = selectResourceCellExitCases(
                  effect->second.resourceCellCases, refinement->second.constant,
                  !refinement->second.equalOnTrue, localResourceCells);
            }
          }
          auto condition = integerValues.find(branch.getCondition());
          if (condition != integerValues.end() &&
              !ambiguousIntegers.contains(branch.getCondition())) {
            bool takeTrue = condition->second != 0;
            mergeSuccessor(
                takeTrue ? branch.getTrueDest() : branch.getFalseDest(),
                takeTrue ? branch.getTrueDestOperands()
                         : branch.getFalseDestOperands(),
                facts, worklist, queued, reached, block, localMemory,
                memoryInputs, edgeMemory, eligibleCells, objectValues,
                integerValues, ambiguousIntegers, handleFieldValues,
                takeTrue ? trueViews : falseViews, viewInputs, edgeViews,
                takeTrue ? truePointers : falsePointers, pointerInputs,
                edgePointers, takeTrue ? trueResourceCells : falseResourceCells,
                resourceCellInputs, edgeResourceCells,
                takeTrue ? trueStatusConstraints : falseStatusConstraints,
                statusConstraintInputs, edgeStatusConstraints,
                edgeIntegerArguments);
            continue;
          }
          if (trueFeasible)
            mergeSuccessor(
                branch.getTrueDest(), branch.getTrueDestOperands(), facts,
                worklist, queued, reached, block, localMemory, memoryInputs,
                edgeMemory, eligibleCells, objectValues, integerValues,
                ambiguousIntegers, handleFieldValues, trueViews, viewInputs,
                edgeViews, truePointers, pointerInputs, edgePointers,
                trueResourceCells, resourceCellInputs, edgeResourceCells,
                trueStatusConstraints, statusConstraintInputs,
                edgeStatusConstraints, edgeIntegerArguments);
          if (falseFeasible)
            mergeSuccessor(
                branch.getFalseDest(), branch.getFalseDestOperands(), facts,
                worklist, queued, reached, block, localMemory, memoryInputs,
                edgeMemory, eligibleCells, objectValues, integerValues,
                ambiguousIntegers, handleFieldValues, falseViews, viewInputs,
                edgeViews, falsePointers, pointerInputs, edgePointers,
                falseResourceCells, resourceCellInputs, edgeResourceCells,
                falseStatusConstraints, statusConstraintInputs,
                edgeStatusConstraints, edgeIntegerArguments);
        } else if (auto branch = dyn_cast<cf::SwitchOp>(operation)) {
          auto selector = integerValues.find(branch.getFlag());
          if (selector != integerValues.end() &&
              !ambiguousIntegers.contains(branch.getFlag())) {
            Block *destination = branch.getDefaultDestination();
            ValueRange operands = branch.getDefaultOperands();
            auto integer = dyn_cast<IntegerType>(branch.getFlag().getType());
            auto caseValues = branch.getCaseValues();
            if (integer && integer.getWidth() <= 64 && caseValues) {
              llvm::APInt selectorBits(integer.getWidth(),
                                       static_cast<uint64_t>(selector->second),
                                       true);
              for (auto [index, caseValue] :
                   llvm::enumerate(caseValues->getValues<llvm::APInt>())) {
                if (selectorBits != caseValue)
                  continue;
                destination = branch.getCaseDestinations()[index];
                operands = branch.getCaseOperands(index);
                break;
              }
            }
            mergeSuccessor(destination, operands, facts, worklist, queued,
                           reached, block, localMemory, memoryInputs,
                           edgeMemory, eligibleCells, objectValues,
                           integerValues, ambiguousIntegers, handleFieldValues,
                           localViews, viewInputs, edgeViews, localPointerCells,
                           pointerInputs, edgePointers, localResourceCells,
                           resourceCellInputs, edgeResourceCells,
                           localStatusConstraints, statusConstraintInputs,
                           edgeStatusConstraints, edgeIntegerArguments);
            continue;
          }
          mergeSuccessor(
              branch.getDefaultDestination(), branch.getDefaultOperands(),
              facts, worklist, queued, reached, block, localMemory,
              memoryInputs, edgeMemory, eligibleCells, objectValues,
              integerValues, ambiguousIntegers, handleFieldValues, localViews,
              viewInputs, edgeViews, localPointerCells, pointerInputs,
              edgePointers, localResourceCells, resourceCellInputs,
              edgeResourceCells, localStatusConstraints, statusConstraintInputs,
              edgeStatusConstraints, edgeIntegerArguments);
          for (auto [index, destination] :
               llvm::enumerate(branch.getCaseDestinations()))
            mergeSuccessor(
                destination, branch.getCaseOperands(index), facts, worklist,
                queued, reached, block, localMemory, memoryInputs, edgeMemory,
                eligibleCells, objectValues, integerValues, ambiguousIntegers,
                handleFieldValues, localViews, viewInputs, edgeViews,
                localPointerCells, pointerInputs, edgePointers,
                localResourceCells, resourceCellInputs, edgeResourceCells,
                localStatusConstraints, statusConstraintInputs,
                edgeStatusConstraints, edgeIntegerArguments);
        } else if (auto returned = dyn_cast<func::ReturnOp>(operation)) {
          ResourceCellExitCases &resourceCellExitCases =
              blockResourceCellExitCases[block];
          PointerExitCases &pointerExitCases = blockPointerExitCases[block];
          ViewExitCases &viewExitCases = blockViewExitCases[block];
          resourceCellExitCases = {};
          pointerExitCases = {};
          viewExitCases = {};
          bool exported = hasConfiguredRoots
                              ? function->hasAttr("vrc.export_id")
                              : function->hasAttr("udoncl.export");
          for (auto [index, operand] :
               llvm::enumerate(returned.getOperands())) {
            Fact fact = valueFact(operand, facts);
            if (exported && fact.resource) {
              returned.emitError(
                  "static device resource escapes through an exported result");
              invalid = true;
            }
            returns[index] = joinFacts(returns[index], fact);
            if (isHandleType(operand.getType())) {
              ObjectRefSet returnedObject =
                  objectAlternative(operand, facts, objectValues);
              if (!objectReturnSeen[index]) {
                objectReturnSeen[index] = true;
                objectReturns[index] = std::move(returnedObject);
              } else {
                objectReturns[index] = joinObjectRefs(
                    std::move(*objectReturns[index]), returnedObject);
              }
            }
            std::optional<int64_t> returnedInteger;
            auto integer = integerValues.find(operand);
            if (integer != integerValues.end())
              returnedInteger = integer->second;
            if (!integerReturnSeen[index]) {
              integerReturnSeen[index] = true;
              integerReturns[index] = returnedInteger;
            } else if (integerReturns[index] != returnedInteger) {
              integerReturns[index] = std::nullopt;
            }
            auto handleField = handleFieldValues.find(operand);
            HandleFieldOriginSet returnedHandleField;
            if (handleField != handleFieldValues.end())
              returnedHandleField = handleField->second;
            if (!handleFieldReturnSeen[index]) {
              handleFieldReturnSeen[index] = true;
              handleFieldReturns[index] = std::move(returnedHandleField);
            } else {
              if (returnedHandleField.refs.empty() &&
                  !returnedHandleField.unknown)
                returnedHandleField.unknown = true;
              if (handleFieldReturns[index].refs.empty() &&
                  !handleFieldReturns[index].unknown)
                handleFieldReturns[index].unknown = true;
              handleFieldReturns[index] = joinHandleFieldOrigins(
                  std::move(handleFieldReturns[index]), returnedHandleField);
            }
          }
          if (!sawReturn) {
            returnedPointerCells = localPointerCells;
            returnedResourceCells = localResourceCells;
            returnedViews = localViews;
            sawReturn = true;
          } else {
            returnedPointerCells =
                joinPointerMaps(returnedPointerCells, localPointerCells,
                                zeroInitializedGlobalObjectIds);
            returnedViews = joinViewMaps(returnedViews, localViews);
            returnedResourceCells =
                joinResourceCellMaps(returnedResourceCells, localResourceCells);
          }
          if (returned.getNumOperands() == 1 &&
              function.getFunctionType().getNumResults() == 1 &&
              function.getFunctionType().getResult(0).isIntOrIndex()) {
            Value status = returned.getOperand(0);
            if (PendingMemoryEffect *pending =
                    pendingEffectAtTerminator(status, block, &operation)) {
              for (const auto &[value, cells] :
                   pending->resourceCellCases.exact)
                addResourceCellExitCase(resourceCellExitCases, value, cells);
              for (const auto &unknown : pending->resourceCellCases.unknown)
                addResourceCellExitCase(resourceCellExitCases, std::nullopt,
                                        unknown.cells, unknown.excluded);
              for (const auto &[value, pointerCells] :
                   pending->pointerCases.exact)
                addPointerExitCase(pointerExitCases, value, pointerCells,
                                   zeroInitializedGlobalObjectIds);
              for (const auto &unknown : pending->pointerCases.unknown)
                addPointerExitCase(
                    pointerExitCases, std::nullopt, unknown.pointerCells,
                    zeroInitializedGlobalObjectIds, unknown.excluded);
              for (const auto &[value, views] : pending->viewCases.exact)
                addViewExitCase(viewExitCases, value, views);
              for (const auto &unknown : pending->viewCases.unknown)
                addViewExitCase(viewExitCases, std::nullopt, unknown.views,
                                unknown.excluded);
            } else if (auto integer = integerValues.find(status);
                       integer != integerValues.end()) {
              addPointerExitCase(pointerExitCases, integer->second,
                                 localPointerCells,
                                 zeroInitializedGlobalObjectIds);
              addViewExitCase(viewExitCases, integer->second, localViews);
              addResourceCellExitCase(resourceCellExitCases, integer->second,
                                      localResourceCells);
            } else if (auto argument = dyn_cast<BlockArgument>(status);
                       argument && argument.getOwner() == block &&
                       llvm::hasSingleElement(*block)) {
              for (Block *predecessor : block->getPredecessors()) {
                Value incoming = successorOperand(predecessor, block,
                                                  argument.getArgNumber());
                auto incomingPending = pendingMemoryEffects.find(incoming);
                if (incomingPending != pendingMemoryEffects.end() &&
                    incomingPending->second.call &&
                    incomingPending->second.call->getBlock() == predecessor &&
                    incomingPending->second.call->getNextNode() ==
                        predecessor->getTerminator()) {
                  for (const auto &[value, cells] :
                       incomingPending->second.resourceCellCases.exact)
                    addResourceCellExitCase(resourceCellExitCases, value,
                                            cells);
                  for (const auto &unknown :
                       incomingPending->second.resourceCellCases.unknown)
                    addResourceCellExitCase(resourceCellExitCases, std::nullopt,
                                            unknown.cells, unknown.excluded);
                  for (const auto &[value, pointers] :
                       incomingPending->second.pointerCases.exact)
                    addPointerExitCase(pointerExitCases, value, pointers,
                                       zeroInitializedGlobalObjectIds);
                  for (const auto &unknown :
                       incomingPending->second.pointerCases.unknown)
                    addPointerExitCase(
                        pointerExitCases, std::nullopt, unknown.pointerCells,
                        zeroInitializedGlobalObjectIds, unknown.excluded);
                  for (const auto &[value, views] :
                       incomingPending->second.viewCases.exact)
                    addViewExitCase(viewExitCases, value, views);
                  for (const auto &unknown :
                       incomingPending->second.viewCases.unknown)
                    addViewExitCase(viewExitCases, std::nullopt, unknown.views,
                                    unknown.excluded);
                  continue;
                }
                auto integers = edgeIntegerArguments.find({predecessor, block});
                auto pointers = edgePointers.find({predecessor, block});
                auto views = edgeViews.find({predecessor, block});
                auto resourceCells =
                    edgeResourceCells.find({predecessor, block});
                auto constraints =
                    edgeStatusConstraints.find({predecessor, block});
                if (pointers == edgePointers.end() &&
                    views == edgeViews.end() &&
                    resourceCells == edgeResourceCells.end())
                  continue;
                std::optional<int64_t> value;
                if (integers != edgeIntegerArguments.end() &&
                    argument.getArgNumber() < integers->second.size())
                  value = integers->second[argument.getArgNumber()];
                StatusExclusions excluded;
                if (constraints != edgeStatusConstraints.end()) {
                  auto found = constraints->second.find(argument);
                  if (found != constraints->second.end())
                    excluded = found->second;
                }
                if (pointers != edgePointers.end())
                  addPointerExitCase(pointerExitCases, value, pointers->second,
                                     zeroInitializedGlobalObjectIds, excluded);
                if (views != edgeViews.end())
                  addViewExitCase(viewExitCases, value, views->second,
                                  excluded);
                if (resourceCells != edgeResourceCells.end())
                  addResourceCellExitCase(resourceCellExitCases, value,
                                          resourceCells->second, excluded);
              }
            } else {
              StatusExclusions excluded;
              auto constraints = localStatusConstraints.find(status);
              if (constraints != localStatusConstraints.end())
                excluded = constraints->second;
              addPointerExitCase(pointerExitCases, std::nullopt,
                                 localPointerCells,
                                 zeroInitializedGlobalObjectIds, excluded);
              addViewExitCase(viewExitCases, std::nullopt, localViews,
                              excluded);
              addResourceCellExitCase(resourceCellExitCases, std::nullopt,
                                      localResourceCells, excluded);
            }
          }
        } else {
          if (operation.getNumRegions() != 0 ||
              (operation.hasTrait<OpTrait::IsTerminator>() &&
               !isa<func::ReturnOp>(operation))) {
            operation.emitError("unsupported region or terminator prevents "
                                "complete device resource provenance");
            invalid = true;
          }
          bool consumesResource =
              llvm::any_of(operation.getOperands(), [&](Value value) {
                return valueFact(value, facts).resource;
              });
          if (consumesResource) {
            operation.emitError(
                "unsupported operation consumes a static device resource: ")
                << operation.getName() << " in @" << function.getSymName();
            invalid = true;
          }
          for (Value result : operation.getResults())
            if (!facts.contains(result))
              updateFact(result, Fact::scalar(), facts, worklist, queued,
                         reached);
        }
        for (auto [index, result] : llvm::enumerate(operation.getResults()))
          recordCase(resultCases, &operation, invocation, index,
                     valueFact(result, facts));
      }
    }
    ResourceCellExitCases resourceCellExitCases;
    PointerExitCases pointerExitCases;
    ViewExitCases viewExitCases;
    for (const auto &[block, cases] : blockResourceCellExitCases) {
      for (const auto &[status, cells] : cases.exact)
        addResourceCellExitCase(resourceCellExitCases, status, cells);
      for (const auto &unknown : cases.unknown)
        addResourceCellExitCase(resourceCellExitCases, std::nullopt,
                                unknown.cells, unknown.excluded);
    }
    for (const auto &[block, cases] : blockPointerExitCases) {
      for (const auto &[status, cells] : cases.exact)
        addPointerExitCase(pointerExitCases, status, cells,
                           zeroInitializedGlobalObjectIds);
      for (const auto &unknown : cases.unknown)
        addPointerExitCase(pointerExitCases, std::nullopt, unknown.pointerCells,
                           zeroInitializedGlobalObjectIds, unknown.excluded);
    }
    for (const auto &[block, cases] : blockViewExitCases) {
      for (const auto &[status, views] : cases.exact)
        addViewExitCase(viewExitCases, status, views);
      for (const auto &unknown : cases.unknown)
        addViewExitCase(viewExitCases, std::nullopt, unknown.views,
                        unknown.excluded);
    }
    active.erase(function);
    for (auto [index, fact] : llvm::enumerate(returns))
      recordCase(functionResultCases, function, invocation, index, fact);
    bool localObjectEscapes =
        llvm::any_of(objectReturns,
                     [&](const auto &result) {
                       return result &&
                              referencesLocalObject(*result, localObjects);
                     }) ||
        llvm::any_of(handleFieldReturns,
                     [&](const auto &origins) {
                       return referencesLocalObject(origins, localObjects);
                     }) ||
        referencesLocalObject(returnedPointerCells, localObjects) ||
        referencesLocalObject(returnedViews, localObjects) ||
        referencesLocalObject(pointerExitCases, localObjects) ||
        referencesLocalObject(viewExitCases, localObjects);
    if (localObjectEscapes) {
      function.emitError("address of a local object escapes its function "
                         "during static device resource analysis in @")
          << function.getSymName();
      invalid = true;
    }
    eraseLocalObjectEntries(handles, localObjects);
    eraseLocalObjectEntries(returnedViews, localObjects);
    eraseLocalObjectEntries(returnedResourceCells, localObjects);
    eraseLocalObjectEntries(returnedPointerCells, localObjects);
    eraseLocalObjectEntries(resourceCellExitCases, localObjects);
    eraseLocalObjectEntries(pointerExitCases, localObjects);
    eraseLocalObjectEntries(viewExitCases, localObjects);
    AnalysisResult result{std::move(returns),
                          std::move(integerReturns),
                          std::move(objectReturns),
                          std::move(handleFieldReturns),
                          std::move(handles),
                          std::move(returnedViews),
                          std::move(returnedResourceCells),
                          std::move(returnedPointerCells),
                          std::move(resourceCellExitCases),
                          std::move(pointerExitCases),
                          std::move(viewExitCases)};
    bool scalarTransparent =
        result.handles == incomingHandles && result.views == incomingViews &&
        result.resourceCells == incomingResourceCells &&
        result.pointerCells == incomingPointerCells &&
        result.resourceCellExitCases.exact.empty() &&
        result.resourceCellExitCases.unknown.empty() &&
        result.pointerExitCases.exact.empty() &&
        result.pointerExitCases.unknown.empty() &&
        result.viewExitCases.exact.empty() &&
        result.viewExitCases.unknown.empty() &&
        llvm::none_of(result.results,
                      [](const Fact &fact) { return fact.resource; }) &&
        llvm::all_of(result.objectResults,
                     [](const auto &value) { return !value; }) &&
        llvm::all_of(result.handleFieldResults, [](const auto &origins) {
          return origins.refs.empty() && !origins.unknown;
        });
    bool cacheComplete = localObjects.empty();
    if (collectingGlobalSummaries && !invalid &&
        (cacheComplete || scalarTransparent))
      analysisCache[cacheKey] = {
          SmallVector<Fact>(arguments),
          SmallVector<std::optional<int64_t>>(argumentIntegers),
          SmallVector<std::optional<ObjectRefSet>>(argumentObjects),
          SmallVector<HandleFieldOriginSet>(argumentHandleFields),
          incomingHandles,
          incomingViews,
          incomingResourceCells,
          incomingPointerCells,
          result,
          cacheComplete,
      };
    return result;
  }

  bool updateFact(Value value, const Fact &incoming,
                  llvm::DenseMap<Value, Fact> &facts,
                  SmallVectorImpl<Block *> &worklist,
                  llvm::DenseSet<Block *> &queued,
                  llvm::DenseSet<Block *> &reached) {
    Fact joined = joinFacts(facts[value], incoming);
    if (facts[value] == joined)
      return false;
    facts[value] = joined;
    for (OpOperand &use : value.getUses()) {
      Block *userBlock = use.getOwner()->getBlock();
      if (reached.contains(userBlock) && queued.insert(userBlock).second)
        worklist.push_back(userBlock);
    }
    return true;
  }

  void mergeSuccessor(
      Block *destination, ValueRange operands,
      llvm::DenseMap<Value, Fact> &facts, SmallVectorImpl<Block *> &worklist,
      llvm::DenseSet<Block *> &queued, llvm::DenseSet<Block *> &reached,
      Block *source, const CellMap &sourceMemory,
      llvm::DenseMap<Block *, CellMap> &memoryInputs,
      llvm::DenseMap<std::pair<Block *, Block *>, CellMap> &edgeMemory,
      const llvm::DenseSet<Value> &eligibleCells,
      llvm::DenseMap<Value, ObjectRefSet> &objectValues,
      llvm::DenseMap<Value, int64_t> &integerValues,
      llvm::DenseSet<Value> &ambiguousIntegers,
      llvm::DenseMap<Value, HandleFieldOriginSet> &handleFieldValues,
      const ViewMap &sourceViews, llvm::DenseMap<Block *, ViewMap> &viewInputs,
      llvm::DenseMap<std::pair<Block *, Block *>, ViewMap> &edgeViews,
      const PointerCellMap &sourcePointers,
      llvm::DenseMap<Block *, PointerCellMap> &pointerInputs,
      llvm::DenseMap<std::pair<Block *, Block *>, PointerCellMap> &edgePointers,
      const ResourceCellMap &sourceResourceCells,
      llvm::DenseMap<Block *, ResourceCellMap> &resourceCellInputs,
      llvm::DenseMap<std::pair<Block *, Block *>, ResourceCellMap>
          &edgeResourceCells,
      const StatusConstraintMap &sourceStatusConstraints,
      llvm::DenseMap<Block *, StatusConstraintMap> &statusConstraintInputs,
      llvm::DenseMap<std::pair<Block *, Block *>, StatusConstraintMap>
          &edgeStatusConstraints,
      llvm::DenseMap<std::pair<Block *, Block *>,
                     SmallVector<std::optional<int64_t>>>
          &edgeIntegerArguments) {
    bool changed = false;
    SmallVector<std::optional<int64_t>> incomingIntegers;
    for (auto [argument, operand] :
         llvm::zip(destination->getArguments(), operands)) {
      if (updateFact(argument, valueFact(operand, facts), facts, worklist,
                     queued, reached)) {
        changed = true;
      }
      if (isHandleType(argument.getType())) {
        ObjectRefSet incomingObject =
            objectAlternative(operand, facts, objectValues);
        auto existingObject = objectValues.find(argument);
        if (existingObject == objectValues.end())
          objectValues[argument] = std::move(incomingObject);
        else
          existingObject->second =
              joinObjectRefs(std::move(existingObject->second), incomingObject);
      }
      auto incomingInteger = integerValues.find(operand);
      bool incomingIntegerIsExact = incomingInteger != integerValues.end() &&
                                    !ambiguousIntegers.contains(operand);
      incomingIntegers.push_back(
          !incomingIntegerIsExact
              ? std::nullopt
              : std::optional<int64_t>(incomingInteger->second));
      if (!ambiguousIntegers.contains(argument)) {
        auto existingInteger = integerValues.find(argument);
        if (!incomingIntegerIsExact) {
          if (existingInteger != integerValues.end()) {
            integerValues.erase(existingInteger);
          }
          ambiguousIntegers.insert(argument);
          changed = true;
        } else if (existingInteger == integerValues.end()) {
          integerValues[argument] = incomingInteger->second;
          changed = true;
        } else if (existingInteger->second != incomingInteger->second) {
          integerValues.erase(existingInteger);
          ambiguousIntegers.insert(argument);
          changed = true;
        }
      }
      auto incomingHandleField = handleFieldValues.find(operand);
      auto existingHandleField = handleFieldValues.find(argument);
      HandleFieldOriginSet previousHandleField =
          existingHandleField == handleFieldValues.end()
              ? HandleFieldOriginSet()
              : existingHandleField->second;
      if (incomingHandleField == handleFieldValues.end()) {
        HandleFieldOriginSet &origins = handleFieldValues[argument];
        origins.unknown = true;
      } else if (existingHandleField == handleFieldValues.end()) {
        handleFieldValues[argument] = incomingHandleField->second;
      } else {
        existingHandleField->second =
            joinHandleFieldOrigins(std::move(existingHandleField->second),
                                   incomingHandleField->second);
      }
      if (!(previousHandleField == handleFieldValues[argument]))
        changed = true;
    }
    edgeIntegerArguments[{source, destination}] = std::move(incomingIntegers);
    StatusConstraintMap incomingStatusConstraints = sourceStatusConstraints;
    for (auto [argument, operand] :
         llvm::zip(destination->getArguments(), operands)) {
      auto incoming = sourceStatusConstraints.find(operand);
      if (incoming == sourceStatusConstraints.end())
        incomingStatusConstraints.erase(argument);
      else
        incomingStatusConstraints[argument] = incoming->second;
    }
    edgeStatusConstraints[{source, destination}] =
        std::move(incomingStatusConstraints);
    StatusConstraintMap recomputedStatusConstraints;
    bool hasStatusConstraintEdge = false;
    for (Block *predecessor : destination->getPredecessors()) {
      auto edge = edgeStatusConstraints.find({predecessor, destination});
      if (edge == edgeStatusConstraints.end())
        continue;
      if (!hasStatusConstraintEdge) {
        recomputedStatusConstraints = edge->second;
        hasStatusConstraintEdge = true;
        continue;
      }
      SmallVector<Value> constraintsToErase;
      for (auto &entry : recomputedStatusConstraints) {
        Value value = entry.first;
        StatusExclusions &exclusions = entry.second;
        auto incoming = edge->second.find(value);
        if (incoming == edge->second.end()) {
          constraintsToErase.push_back(value);
          continue;
        }
        StatusExclusions intersection;
        std::set_intersection(exclusions.begin(), exclusions.end(),
                              incoming->second.begin(), incoming->second.end(),
                              std::inserter(intersection, intersection.end()));
        if (intersection.empty())
          constraintsToErase.push_back(value);
        else
          exclusions = std::move(intersection);
      }
      for (Value value : constraintsToErase)
        recomputedStatusConstraints.erase(value);
    }
    auto statusConstraintsEqual = [](const StatusConstraintMap &left,
                                     const StatusConstraintMap &right) {
      if (left.size() != right.size())
        return false;
      return llvm::all_of(left, [&](const auto &entry) {
        auto found = right.find(entry.first);
        return found != right.end() && found->second == entry.second;
      });
    };
    if (!statusConstraintsEqual(statusConstraintInputs[destination],
                                recomputedStatusConstraints))
      changed = true;
    statusConstraintInputs[destination] =
        std::move(recomputedStatusConstraints);
    edgeMemory[{source, destination}] = sourceMemory;
    CellMap recomputed;
    for (Value cell : eligibleCells) {
      CellState joined;
      for (Block *predecessor : destination->getPredecessors()) {
        auto edge = edgeMemory.find({predecessor, destination});
        if (edge == edgeMemory.end())
          continue;
        CellState incoming = edge->second.lookup(cell);
        if (!incoming.hasPath)
          incoming = {true, false, Fact()};
        joined = joinCellState(joined, incoming);
      }
      recomputed[cell] = joined;
      if (!(memoryInputs[destination].lookup(cell) == joined))
        changed = true;
    }
    memoryInputs[destination] = std::move(recomputed);
    auto edgeKey = std::make_pair(source, destination);
    auto previousViewEdge = edgeViews.find(edgeKey);
    if (previousViewEdge == edgeViews.end()) {
      edgeViews[edgeKey] = sourceViews;
      ViewMap recomputedViews;
      bool hasViewEdge = false;
      for (Block *predecessor : destination->getPredecessors()) {
        auto edge = edgeViews.find({predecessor, destination});
        if (edge == edgeViews.end())
          continue;
        if (!hasViewEdge) {
          recomputedViews = edge->second;
          hasViewEdge = true;
        } else {
          recomputedViews = joinViewMaps(recomputedViews, edge->second);
        }
      }
      if (viewInputs[destination] != recomputedViews)
        changed = true;
      viewInputs[destination] = std::move(recomputedViews);
    } else {
      SmallVector<ObjectRef> changedViews =
          changedObjectRefs(previousViewEdge->second, sourceViews);
      if (!changedViews.empty()) {
        edgeViews[edgeKey] = sourceViews;
        ViewMap &destinationViews = viewInputs[destination];
        for (ObjectRef object : changedViews) {
          ViewState joined;
          bool hasEdge = false;
          bool hasValue = false;
          for (Block *predecessor : destination->getPredecessors()) {
            auto edge = edgeViews.find({predecessor, destination});
            if (edge == edgeViews.end())
              continue;
            auto value = edge->second.find(object);
            hasValue |= value != edge->second.end();
            ViewState incoming =
                value == edge->second.end() ? ViewState{true} : value->second;
            joined = hasEdge ? joinViewStates(joined, incoming) : incoming;
            hasEdge = true;
          }
          auto previous = destinationViews.find(object);
          if (!hasValue) {
            if (previous != destinationViews.end()) {
              destinationViews.erase(object);
              changed = true;
            }
          } else if (previous == destinationViews.end() ||
                     !(previous->second == joined)) {
            destinationViews[object] = joined;
            changed = true;
          }
        }
      }
    }
    auto previousPointerEdge = edgePointers.find(edgeKey);
    if (previousPointerEdge == edgePointers.end()) {
      edgePointers[edgeKey] = sourcePointers;
      PointerCellMap recomputedPointers;
      bool hasPointerEdge = false;
      for (Block *predecessor : destination->getPredecessors()) {
        auto edge = edgePointers.find({predecessor, destination});
        if (edge == edgePointers.end())
          continue;
        if (!hasPointerEdge) {
          recomputedPointers = edge->second;
          hasPointerEdge = true;
        } else {
          recomputedPointers = joinPointerMaps(recomputedPointers, edge->second,
                                               zeroInitializedGlobalObjectIds);
        }
      }
      if (pointerInputs[destination] != recomputedPointers)
        changed = true;
      pointerInputs[destination] = std::move(recomputedPointers);
    } else {
      SmallVector<ObjectRef> changedPointers =
          changedObjectRefs(previousPointerEdge->second, sourcePointers);
      if (!changedPointers.empty()) {
        edgePointers[edgeKey] = sourcePointers;
        PointerCellMap &destinationPointers = pointerInputs[destination];
        for (ObjectRef object : changedPointers) {
          PointerCellState joined;
          bool hasEdge = false;
          bool hasValue = false;
          for (Block *predecessor : destination->getPredecessors()) {
            auto edge = edgePointers.find({predecessor, destination});
            if (edge == edgePointers.end())
              continue;
            auto value = edge->second.find(object);
            hasValue |= value != edge->second.end();
            PointerCellState incoming =
                value == edge->second.end()
                    ? (zeroInitializedGlobalObjectIds.contains(object.base)
                           ? PointerCellState{true, true, {}}
                           : PointerCellState{true, false, {}})
                    : value->second;
            joined = hasEdge ? joinPointerCells(joined, incoming) : incoming;
            hasEdge = true;
          }
          auto previous = destinationPointers.find(object);
          if (!hasValue) {
            if (previous != destinationPointers.end()) {
              destinationPointers.erase(object);
              changed = true;
            }
          } else if (previous == destinationPointers.end() ||
                     !(previous->second == joined)) {
            destinationPointers[object] = joined;
            changed = true;
          }
        }
      }
    }
    auto previousResourceCellEdge = edgeResourceCells.find(edgeKey);
    if (previousResourceCellEdge == edgeResourceCells.end()) {
      edgeResourceCells[edgeKey] = sourceResourceCells;
      ResourceCellMap recomputedResourceCells;
      bool hasResourceCellEdge = false;
      for (Block *predecessor : destination->getPredecessors()) {
        auto edge = edgeResourceCells.find({predecessor, destination});
        if (edge == edgeResourceCells.end())
          continue;
        if (!hasResourceCellEdge) {
          recomputedResourceCells = edge->second;
          hasResourceCellEdge = true;
        } else {
          recomputedResourceCells =
              joinResourceCellMaps(recomputedResourceCells, edge->second);
        }
      }
      if (resourceCellInputs[destination] != recomputedResourceCells)
        changed = true;
      resourceCellInputs[destination] = std::move(recomputedResourceCells);
    } else {
      SmallVector<ObjectRef> changedResourceCells = changedObjectRefs(
          previousResourceCellEdge->second, sourceResourceCells);
      if (!changedResourceCells.empty()) {
        edgeResourceCells[edgeKey] = sourceResourceCells;
        ResourceCellMap &destinationCells = resourceCellInputs[destination];
        for (ObjectRef object : changedResourceCells) {
          CellState joined;
          bool hasValue = false;
          for (Block *predecessor : destination->getPredecessors()) {
            auto edge = edgeResourceCells.find({predecessor, destination});
            if (edge == edgeResourceCells.end())
              continue;
            auto value = edge->second.find(object);
            CellState incoming = value == edge->second.end()
                                     ? CellState{true, false, Fact()}
                                     : value->second;
            hasValue |= value != edge->second.end();
            joined = joinCellState(joined, incoming);
          }
          auto previous = destinationCells.find(object);
          if (!hasValue) {
            if (previous != destinationCells.end()) {
              destinationCells.erase(object);
              changed = true;
            }
          } else if (previous == destinationCells.end() ||
                     !(previous->second == joined)) {
            destinationCells[object] = joined;
            changed = true;
          }
        }
      }
    }
    bool firstReach = reached.insert(destination).second;
    if ((changed || firstReach) && queued.insert(destination).second)
      worklist.push_back(destination);
  }

  DictionaryAttr caseAttribute(const Case &entry) {
    SmallVector<Attribute> ids;
    for (int32_t id : entry.fact.ids)
      ids.push_back(builder->getI32IntegerAttr(id));
    return builder->getDictionaryAttr({
        builder->getNamedAttr("call_path", entry.invocation.path),
        builder->getNamedAttr("may_be_null",
                              builder->getBoolAttr(entry.fact.mayBeNull)),
        builder->getNamedAttr("resource_ids", builder->getArrayAttr(ids)),
        builder->getNamedAttr("root", entry.invocation.root),
        builder->getNamedAttr("unknown",
                              builder->getBoolAttr(entry.fact.unknown)),
    });
  }

  void emitCases(CaseMap &map, StringRef name) {
    for (auto &[operation, cases] : map) {
      llvm::sort(cases, [&](const Case &left, const Case &right) {
        if (left.index != right.index)
          return left.index < right.index;
        if (left.invocation.root.getValue() != right.invocation.root.getValue())
          return left.invocation.root.getValue() <
                 right.invocation.root.getValue();
        return invocationPathKey(left.invocation.path) <
               invocationPathKey(right.invocation.path);
      });
      SmallVector<Attribute> indexed;
      for (auto begin = cases.begin(); begin != cases.end();) {
        auto end = std::find_if(begin, cases.end(), [&](const Case &entry) {
          return entry.index != begin->index;
        });
        SmallVector<Attribute> records;
        for (const Case &entry : llvm::make_range(begin, end))
          records.push_back(caseAttribute(entry));
        indexed.push_back(builder->getDictionaryAttr({
            builder->getNamedAttr("index",
                                  builder->getI32IntegerAttr(begin->index)),
            builder->getNamedAttr("cases", builder->getArrayAttr(records)),
        }));
        begin = end;
      }
      operation->setAttr(name, builder->getArrayAttr(indexed));
    }
  }

  void runOnOperation() override {
    module = getOperation();
    debugProgress = module->hasAttr(debugProvenanceAttribute);
    hasConfiguredRoots =
        llvm::any_of(module.getOps<func::FuncOp>(), [](func::FuncOp function) {
          return !function.isExternal() && function->hasAttr("vrc.export_id");
        });
    Builder localBuilder(module.getContext());
    builder = &localBuilder;
    auto discovery = module->getAttrOfType<DictionaryAttr>(discoveryAttribute);
    auto version =
        discovery
            ? dyn_cast_or_null<IntegerAttr>(discovery.get("schema_version"))
            : IntegerAttr();
    contexts = discovery
                   ? dyn_cast_or_null<ArrayAttr>(discovery.get("contexts"))
                   : ArrayAttr();
    if (!version || version.getInt() != 1 || !contexts) {
      module.emitError("static device resource provenance requires discovery "
                       "schema version 1");
      signalPassFailure();
      return;
    }
    module.walk([&](Operation *operation) {
      operation->removeAttr(operandAttribute);
      operation->removeAttr(resultAttribute);
      operation->removeAttr(argumentAttribute);
      operation->removeAttr(functionResultAttribute);
    });
    module->removeAttr(provenanceAttribute);
    functions.clear();
    eligibleGlobals.clear();
    eligibleHandleGlobals.clear();
    globalObjects.clear();
    globalHandleObjects.clear();
    globalObjectIds.clear();
    globalHandleObjectIds.clear();
    zeroInitializedGlobalObjectIds.clear();
    globalHandleFacts.clear();
    globalHandleWrites.clear();
    globalViewFacts.clear();
    globalViewWrites.clear();
    globalPointerFacts.clear();
    globalPointerWrites.clear();
    localHandleObjects.clear();
    objectSizes.clear();
    restrictedBufferViewLayouts.clear();
    nextObjectId = 0;
    globalFacts.clear();
    globalWrites.clear();
    resourceGlobals.clear();
    operandCases.clear();
    resultCases.clear();
    argumentCases.clear();
    functionResultCases.clear();
    active.clear();
    invalid = false;
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      auto found = functions.find(function.getSymName());
      if (found == functions.end()) {
        functions.try_emplace(function.getSymName(), function);
      } else if (found->getValue().isExternal() && !function.isExternal()) {
        found->getValue() = function;
      }
    }
    DataLayout layout(module);
    for (auto global : module.getOps<::vrc::udonhl::GlobalOp>()) {
      auto initial =
          dyn_cast_or_null<IntegerAttr>(global.getInitialValueAttr());
      auto bytes =
          dyn_cast_or_null<DenseI8ArrayAttr>(global.getInitialValueAttr());
      bool zeroInitialized =
          (initial && initial.getInt() == 0) ||
          (bytes && llvm::all_of(bytes.asArrayRef(),
                                 [](int8_t byte) { return byte == 0; }));
      auto pointer = dyn_cast<::vrc::udonhl::PtrType>(global.getStorageType());
      if (global.getLinkage() == "internal" && !global.getConstant() &&
          global.getAddressSpace() == 0 && global.getAlignment() == 4 &&
          pointer && pointer.getAddressSpace() == 0 && initial &&
          initial.getInt() == 0)
        eligibleGlobals.try_emplace(global.getSymName(), global);

      auto array = dyn_cast<LLVM::LLVMArrayType>(global.getStorageType());
      auto structure = dyn_cast<LLVM::LLVMStructType>(global.getStorageType());
      bool handleStructure = structure && structure.getBody().size() == 2 &&
                             llvm::all_of(structure.getBody(), [](Type type) {
                               return type.isInteger(32);
                             });
      uint64_t handleBytes = array && array.getElementType().isInteger(8)
                                 ? array.getNumElements()
                                 : (handleStructure ? 8 : 0);
      auto objectBytes = fixedObjectBytes(layout, global.getStorageType());
      if (global.getLinkage() == "internal" && !global.getConstant() &&
          global.getAddressSpace() == 0 && succeeded(objectBytes) &&
          *objectBytes > 0 &&
          *objectBytes <= std::numeric_limits<uint32_t>::max()) {
        globalObjects[global.getSymName()] = nextObjectId++;
        objectSizes[globalObjects[global.getSymName()]] =
            static_cast<uint32_t>(*objectBytes);
        globalObjectIds.insert(globalObjects[global.getSymName()]);
        if (auto layout = global->getAttrOfType<DictionaryAttr>(
                bufferViewLayoutAttribute)) {
          auto version = layout.getAs<IntegerAttr>("schema_version");
          auto count = layout.getAs<IntegerAttr>("count");
          auto stride = layout.getAs<IntegerAttr>("stride_bytes");
          auto offsets = layout.getAs<DenseI32ArrayAttr>("offsets");
          auto isI32 = [](IntegerAttr value) {
            return value && value.getType().isInteger(32);
          };
          if (!isI32(version) || version.getInt() != 1 || !isI32(count) ||
              !isI32(stride) || !offsets || count.getInt() < 0 ||
              stride.getInt() < 0 ||
              (count.getInt() == 0 &&
               (stride.getInt() != 0 || !offsets.empty())) ||
              (count.getInt() > 0 &&
               (stride.getInt() == 0 || offsets.empty()))) {
            global.emitError("malformed buffer view layout metadata");
            invalid = true;
          } else {
            BufferViewLayout parsed{static_cast<uint32_t>(count.getInt()),
                                    static_cast<uint32_t>(stride.getInt()),
                                    {}};
            for (int32_t offset : offsets.asArrayRef()) {
              uint64_t strideBytes = parsed.stride;
              uint64_t viewBytes = UDONCL_UDON_ABI_BUFFER_VIEW_BYTES;
              if (offset < 0 || strideBytes < viewBytes ||
                  static_cast<uint64_t>(offset) > strideBytes - viewBytes) {
                global.emitError("buffer view layout offset is out of bounds");
                invalid = true;
                continue;
              }
              uint64_t lastStart =
                  static_cast<uint64_t>(parsed.count - 1) * strideBytes +
                  static_cast<uint64_t>(offset);
              if (lastStart > std::numeric_limits<int32_t>::max() ||
                  lastStart + viewBytes > *objectBytes) {
                global.emitError("buffer view layout extent is out of bounds");
                invalid = true;
                continue;
              }
              parsed.memberOffsets.push_back(offset);
            }
            llvm::sort(parsed.memberOffsets);
            if (std::adjacent_find(parsed.memberOffsets.begin(),
                                   parsed.memberOffsets.end()) !=
                parsed.memberOffsets.end()) {
              global.emitError("buffer view layout offsets are not unique");
              invalid = true;
            }
            restrictedBufferViewLayouts[globalObjects[global.getSymName()]] =
                std::move(parsed);
          }
        }
        if (zeroInitialized)
          zeroInitializedGlobalObjectIds.insert(
              globalObjects[global.getSymName()]);
      }
      if (global.getLinkage() == "internal" && !global.getConstant() &&
          global.getAddressSpace() == 0 && global.getAlignment() >= 4 &&
          handleBytes >= 8 && handleBytes % 8 == 0 && bytes &&
          bytes.size() == handleBytes &&
          llvm::all_of(bytes.asArrayRef(),
                       [](int8_t byte) { return byte == 0; })) {
        eligibleHandleGlobals.try_emplace(global.getSymName(), global);
        globalHandleObjects[global.getSymName()] =
            globalObjects[global.getSymName()];
        ObjectId object = globalHandleObjects[global.getSymName()];
        globalHandleObjectIds.insert(object);
        objectSizes[object] = static_cast<uint32_t>(handleBytes);
      }
    }
    module.walk([&](::vrc::udonhl::AddressOfOp address) {
      StringRef name = address.getGlobalAttr().getValue();
      if (!eligibleGlobals.contains(name))
        return;
      if (!llvm::all_of(address.getResult().getUses(), [&](OpOperand &use) {
            return isExactCellAccess(use, address.getResult());
          }))
        eligibleGlobals.erase(name);
    });
    for (auto &[name, global] : eligibleGlobals)
      globalFacts[name] = Fact::scalar();
    SmallVector<func::FuncOp> roots;
    for (func::FuncOp function : module.getOps<func::FuncOp>())
      if (!function.isExternal() &&
          (hasConfiguredRoots ? function->hasAttr("vrc.export_id")
                              : function->hasAttr("udoncl.export")))
        roots.push_back(function);
    llvm::sort(roots, [](func::FuncOp left, func::FuncOp right) {
      return left.getSymName() < right.getSymName();
    });
    bool converged = false;
    collectingGlobalSummaries = true;
    for (unsigned iteration = 0; iteration != 100 && !invalid; ++iteration) {
      analysisCache.clear();
      globalWrites.clear();
      globalHandleWrites.clear();
      globalViewWrites.clear();
      globalPointerWrites.clear();
      for (auto &[name, global] : eligibleGlobals)
        globalWrites[name] = Fact();
      for (func::FuncOp root : roots) {
        auto rootStarted = std::chrono::steady_clock::now();
        uint64_t callsBeforeRoot = analysisCalls;
        if (debugProgress)
          llvm::errs() << "udoncl-provenance-root=" << root.getSymName()
                       << " iteration=" << iteration << " state=start\n";
        SmallVector<Fact> arguments(root.getNumArguments(), Fact::scalar());
        SmallVector<std::optional<int64_t>> argumentIntegers(
            root.getNumArguments(), std::nullopt);
        SmallVector<std::optional<ObjectRefSet>> argumentObjects(
            root.getNumArguments(), std::nullopt);
        SmallVector<HandleFieldOriginSet> argumentHandleFields(
            root.getNumArguments());
        HandleMap handles;
        ViewMap views = globalViewFacts;
        ResourceCellMap resourceCells;
        PointerCellMap pointerCells = globalPointerFacts;
        handles = globalHandleFacts;
        AnalysisResult result = analyze(
            root,
            {FlatSymbolRefAttr::get(module.getContext(), root.getSymName()),
             builder->getArrayAttr({})},
            arguments, argumentIntegers, argumentObjects, argumentHandleFields,
            handles, views, resourceCells, pointerCells);
        if (debugProgress) {
          auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - rootStarted)
                             .count();
          llvm::errs() << "udoncl-provenance-root=" << root.getSymName()
                       << " iteration=" << iteration
                       << " state=complete elapsed_ms=" << elapsed
                       << " analyses=" << (analysisCalls - callsBeforeRoot)
                       << "\n";
        }
        for (const auto &[handle, fact] : result.handles)
          if (isGlobalHandle(handle))
            globalHandleWrites[handle] =
                joinFacts(globalHandleWrites[handle], fact);
        for (const auto &[cell, state] : result.pointerCells) {
          if (!globalObjectIds.contains(cell.base) || !state.hasPath ||
              (state.objects.refs.empty() && !state.objects.unknown))
            continue;
          ObjectRefSet persistentTargets;
          persistentTargets.unknown = state.objects.unknown;
          for (ObjectRef target : state.objects.refs)
            if (globalObjectIds.contains(target.base))
              persistentTargets.refs.push_back(target);
            else
              persistentTargets.unknown = true;
          PointerCellState &write = globalPointerWrites[cell];
          if (!write.hasPath) {
            write = state;
            write.objects = std::move(persistentTargets);
            write.initializedOnAllPaths = true;
          } else {
            write.objects =
                joinObjectRefs(std::move(write.objects), persistentTargets);
          }
        }
        for (const auto &[view, state] : result.views)
          recordGlobalViewWrite(view, state);
        if (invalid)
          break;
      }
      if (invalid)
        break;
      converged = true;
      for (auto &[name, global] : eligibleGlobals) {
        Fact next = resourceGlobals.contains(name)
                        ? joinFacts(Fact::nullHandle(), globalWrites[name])
                        : Fact::scalar();
        if (!(globalFacts[name] == next)) {
          globalFacts[name] = next;
          converged = false;
        }
      }
      for (const auto &[handle, write] : globalHandleWrites) {
        Fact next = joinFacts(Fact::nullHandle(), write);
        auto current = globalHandleFacts.find(handle);
        if (current == globalHandleFacts.end() || !(current->second == next)) {
          globalHandleFacts[handle] = next;
          converged = false;
        }
      }
      for (const auto &[cell, write] : globalPointerWrites) {
        PointerCellState next = write;
        auto current = globalPointerFacts.find(cell);
        if (current != globalPointerFacts.end()) {
          next.objects = joinObjectRefs(current->second.objects, next.objects);
          next.hasPath = true;
          next.initializedOnAllPaths = true;
        }
        if (current == globalPointerFacts.end() || !(current->second == next)) {
          globalPointerFacts[cell] = std::move(next);
          converged = false;
        }
      }
      for (const auto &[view, write] : globalViewWrites) {
        auto current = globalViewFacts.find(view);
        ViewState next = current == globalViewFacts.end()
                             ? write
                             : joinViewStates(current->second, write);
        if (current == globalViewFacts.end() || !(current->second == next)) {
          globalViewFacts[view] = std::move(next);
          converged = false;
        }
      }
      if (converged) {
        if (collectingGlobalSummaries) {
          collectingGlobalSummaries = false;
          converged = false;
          continue;
        }
        break;
      }
    }
    if (!invalid && !converged) {
      module.emitError("global device resource provenance did not converge");
      invalid = true;
    }
    if (invalid) {
      signalPassFailure();
      return;
    }
    emitCases(operandCases, operandAttribute);
    emitCases(resultCases, resultAttribute);
    emitCases(argumentCases, argumentAttribute);
    emitCases(functionResultCases, functionResultAttribute);
    module->setAttr(provenanceAttribute,
                    builder->getDictionaryAttr({
                        builder->getNamedAttr("discovery_schema_version",
                                              builder->getI32IntegerAttr(1)),
                        builder->getNamedAttr("schema_version",
                                              builder->getI32IntegerAttr(1)),
                    }));
  }
};

} // namespace
} // namespace vrc
