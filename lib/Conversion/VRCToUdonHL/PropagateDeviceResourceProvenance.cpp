#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <limits>
#include <map>
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

struct ObjectRefSet {
  bool unknown = false;
  SmallVector<ObjectRef> refs;

  static ObjectRefSet singleton(ObjectRef object) {
    ObjectRefSet result;
    result.refs.push_back(object);
    return result;
  }
};

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

using HandleMap = std::map<ObjectRef, Fact>;

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

using ViewMap = std::map<ObjectRef, ViewState>;

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

using PointerCellMap = std::map<ObjectRef, PointerCellState>;

struct PointerExitCases {
  std::map<int64_t, PointerCellMap> exact;
  std::optional<PointerCellMap> unknown;
};

bool operator==(const PointerCellState &left, const PointerCellState &right) {
  return left.hasPath == right.hasPath &&
         left.initializedOnAllPaths == right.initializedOnAllPaths &&
         left.objects == right.objects;
}

struct AnalysisResult {
  SmallVector<Fact> results;
  SmallVector<std::optional<int64_t>> integerResults;
  SmallVector<HandleFieldOriginSet> handleFieldResults;
  HandleMap handles;
  ViewMap views;
  PointerCellMap pointerCells;
  PointerExitCases pointerExitCases;
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
                               const PointerCellMap &right) {
  PointerCellMap joined;
  for (const auto &[object, state] : left) {
    auto found = right.find(object);
    PointerCellState other = found == right.end()
                                 ? PointerCellState{true, false, {}}
                                 : found->second;
    joined[object] = joinPointerCells(state, other);
  }
  for (const auto &[object, state] : right) {
    if (left.count(object))
      continue;
    joined[object] = joinPointerCells(PointerCellState{true, false, {}}, state);
  }
  return joined;
}

void addPointerExitCase(PointerExitCases &cases, std::optional<int64_t> status,
                        const PointerCellMap &pointerCells) {
  constexpr size_t maxExactCases = 16;
  if (!status ||
      (!cases.exact.count(*status) && cases.exact.size() >= maxExactCases)) {
    cases.unknown = cases.unknown
                        ? joinPointerMaps(*cases.unknown, pointerCells)
                        : std::optional<PointerCellMap>(pointerCells);
    return;
  }
  auto found = cases.exact.find(*status);
  if (found == cases.exact.end())
    cases.exact.emplace(*status, pointerCells);
  else
    found->second = joinPointerMaps(found->second, pointerCells);
}

PointerCellMap selectPointerExitCases(const PointerExitCases &cases,
                                      int64_t constant, bool equal,
                                      const PointerCellMap &fallback) {
  PointerCellMap selected;
  bool hasSelected = false;
  for (const auto &[status, pointerCells] : cases.exact) {
    if ((status == constant) != equal)
      continue;
    selected =
        hasSelected ? joinPointerMaps(selected, pointerCells) : pointerCells;
    hasSelected = true;
  }
  if (cases.unknown) {
    selected = hasSelected ? joinPointerMaps(selected, *cases.unknown)
                           : *cases.unknown;
    hasSelected = true;
  }
  return hasSelected ? selected : fallback;
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

bool operator==(const Fact &left, const Fact &right) {
  return left.initialized == right.initialized &&
         left.resource == right.resource && left.unknown == right.unknown &&
         left.mayBeNull == right.mayBeNull && left.zero == right.zero &&
         left.ids == right.ids;
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

struct Invocation {
  FlatSymbolRefAttr root;
  ArrayAttr path;
};

struct PendingPointerEffect {
  Operation *call = nullptr;
  PointerExitCases cases;
};

struct ComparisonRefinement {
  Value subject;
  int64_t constant = 0;
  bool equalOnTrue = true;
};

struct Case {
  Invocation invocation;
  unsigned index;
  Fact fact;
};

using CaseMap = llvm::DenseMap<Operation *, SmallVector<Case>>;

bool sameInvocation(const Invocation &left, const Invocation &right) {
  return left.root == right.root && left.path == right.path;
}

void recordCase(CaseMap &map, Operation *operation,
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

bool isKnownConsumer(StringRef name) {
  return name == "clReleaseMemObject" || name == "clEnqueueWriteBuffer" ||
         name == "clEnqueueReadBuffer" || name == "clEnqueueCopyBuffer" ||
         name == "clSetKernelArg";
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
  HandleMap globalHandleFacts;
  HandleMap globalHandleWrites;
  llvm::DenseMap<ObjectId, uint32_t> objectSizes;
  std::map<std::tuple<std::string, std::string, Operation *>, ObjectId>
      localHandleObjects;
  ObjectId nextObjectId = 0;
  bool invalid = false;

  std::string invocationPathKey(ArrayAttr path) {
    std::string storage;
    llvm::raw_string_ostream stream(storage);
    path.print(stream);
    return storage;
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
    return found != objectSizes.end() && object.offset >= 0 &&
           static_cast<uint64_t>(object.offset) + 20 <= found->second;
  }

  bool canHoldHandle(const ObjectRef &object) const {
    auto found = objectSizes.find(object.base);
    return found != objectSizes.end() && object.offset >= 0 &&
           static_cast<uint64_t>(object.offset) + 8 <= found->second;
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
    auto resource = handles.find(slot->handle);
    if (resource == handles.end() || !resource->second.resource ||
        resource->second.unknown)
      return;
    state.resource = resource->second;
    state.initialized = true;
    state.mayBeInvalid = true;
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
    }
    return recognized;
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

  void updateHandle(ObjectRef object, const Fact &incoming, HandleMap &handles,
                    SmallVectorImpl<Block *> &worklist,
                    llvm::DenseSet<Block *> &queued,
                    const llvm::DenseSet<Block *> &reached) {
    Fact joined = joinFacts(handles[object], incoming);
    if (handles[object] == joined)
      return;
    handles[object] = joined;
    for (Block *block : reached)
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
      call.emitError("resource use has unresolved non-handle provenance");
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
                         const PointerCellMap &incomingPointerCells) {
    if (!active.insert(function).second) {
      if (llvm::none_of(arguments,
                        [](const Fact &fact) { return fact.resource; }))
        return {SmallVector<Fact>(function.getFunctionType().getNumResults(),
                                  Fact::scalar()),
                SmallVector<std::optional<int64_t>>(
                    function.getFunctionType().getNumResults(), std::nullopt),
                SmallVector<HandleFieldOriginSet>(
                    function.getFunctionType().getNumResults()),
                incomingHandles,
                incomingViews,
                incomingPointerCells,
                {}};
      function.emitError(
          "resource provenance through recursion is unsupported");
      invalid = true;
      return {SmallVector<Fact>(),
              SmallVector<std::optional<int64_t>>(),
              SmallVector<HandleFieldOriginSet>(),
              HandleMap(),
              ViewMap(),
              PointerCellMap(),
              {}};
    }
    llvm::DenseMap<Value, Fact> facts;
    llvm::DenseMap<Value, int64_t> integerValues;
    llvm::DenseSet<Value> ambiguousIntegers;
    llvm::DenseMap<Value, ObjectRefSet> objectValues;
    llvm::DenseMap<Value, HandleFieldOriginSet> handleFieldValues;
    HandleMap handles = incomingHandles;
    llvm::DenseSet<Value> eligibleCells;
    function.walk([&](::vrc::udonhl::AllocaOp allocation) {
      if (isSingletonCell(allocation))
        eligibleCells.insert(allocation.getResult());
      auto bytes = constantAllocationBytes(allocation);
      if (bytes && (*bytes == 8 || *bytes >= 20)) {
        ObjectId object = localObject(invocation, allocation, *bytes);
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
    SmallVector<HandleFieldOriginSet> handleFieldReturns(
        function.getFunctionType().getNumResults());
    SmallVector<bool> handleFieldReturnSeen(
        function.getFunctionType().getNumResults(), false);
    SmallVector<bool> integerReturnSeen(
        function.getFunctionType().getNumResults(), false);
    SmallVector<Block *> worklist;
    llvm::DenseSet<Block *> queued;
    llvm::DenseSet<Block *> reached;
    llvm::DenseMap<Block *, CellMap> memoryInputs;
    llvm::DenseMap<std::pair<Block *, Block *>, CellMap> edgeMemory;
    llvm::DenseMap<Block *, PointerCellMap> pointerInputs;
    llvm::DenseMap<std::pair<Block *, Block *>, PointerCellMap> edgePointers;
    llvm::DenseMap<std::pair<Block *, Block *>,
                   SmallVector<std::optional<int64_t>>>
        edgeIntegerArguments;
    llvm::DenseMap<Block *, ViewMap> viewInputs;
    llvm::DenseMap<std::pair<Block *, Block *>, ViewMap> edgeViews;
    llvm::DenseMap<Value, PendingPointerEffect> pendingPointerEffects;
    llvm::DenseMap<Value, ComparisonRefinement> comparisonRefinements;
    PointerCellMap returnedPointerCells;
    ViewMap returnedViews;
    PointerExitCases pointerExitCases;
    bool sawReturn = false;
    llvm::DenseSet<Value> resourceCells;
    if (!function.empty()) {
      worklist.push_back(&function.front());
      queued.insert(&function.front());
      reached.insert(&function.front());
      pointerInputs[&function.front()] = incomingPointerCells;
      viewInputs[&function.front()] = incomingViews;
      for (Value cell : eligibleCells)
        memoryInputs[&function.front()][cell] = {true, false, Fact()};
    }
    unsigned iterations = 0;
    while (!worklist.empty() && !invalid) {
      if (++iterations > 10000) {
        function.emitError("device resource provenance did not converge");
        invalid = true;
        break;
      }
      Block *block = worklist.pop_back_val();
      queued.erase(block);
      CellMap localMemory = memoryInputs[block];
      PointerCellMap localPointerCells = pointerInputs[block];
      ViewMap localViews = viewInputs[block];
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
          bool directViewField = false;
          if (exactCell) {
            auto origins = handleFieldValues.find(store.getValue());
            directViewField = recordDirectViewField(
                *exactCell, store.getValue().getType(),
                origins == handleFieldValues.end() ? nullptr : &origins->second,
                localViews, handles);
          }
          if (exactCell && isHandleType(store.getValue().getType()) &&
              !stored.resource) {
            PointerCellState incoming;
            incoming.hasPath = true;
            incoming.initializedOnAllPaths = true;
            if (storedObjects == objectValues.end())
              incoming.objects.unknown = true;
            else
              incoming.objects = storedObjects->second;
            localPointerCells[*exactCell] = std::move(incoming);
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
          if (auto global = globalCell(cell)) {
            updateFact(load.getResult(), globalFacts[global.getSymName()],
                       facts, worklist, queued, reached);
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
          auto cellObjects = objectValues.find(cell);
          auto exactCell = cellObjects == objectValues.end()
                               ? std::nullopt
                               : exactObject(cellObjects->second);
          if (exactCell && load.getType().isInteger(32)) {
            ObjectRef handle = *exactCell;
            std::optional<HandleFieldKind> kind;
            if (handle.offset >= 4) {
              ObjectRef generation{handle.base, handle.offset - 4};
              if (canHoldHandle(generation) && handles.count(generation)) {
                handle = generation;
                kind = HandleFieldKind::Generation;
              }
            }
            if (!kind && canHoldHandle(handle) && handles.count(handle))
              kind = HandleFieldKind::Slot;
            if (kind) {
              handleFieldValues[load.getResult()] = {false, {{handle, *kind}}};
              if (*kind == HandleFieldKind::Slot)
                updateFact(load.getResult(), handles[handle], facts, worklist,
                           queued, reached);
            }
          }
          if (exactCell && isHandleType(load.getType())) {
            auto pointerCell = localPointerCells.find(*exactCell);
            if (pointerCell != localPointerCells.end() &&
                pointerCell->second.hasPath) {
              if (!pointerCell->second.initializedOnAllPaths) {
                load.emitError("aggregate object pointer is not initialized "
                               "on every reaching path");
                invalid = true;
                continue;
              }
              objectValues[load.getResult()] = pointerCell->second.objects;
            }
          }
        } else if (auto call = dyn_cast<func::CallOp>(operation)) {
          StringRef calleeName = call.getCallee();
          if (calleeName == "udonclResolvePhysicalBuffer") {
            analyzePhysicalBufferResolver(call, facts);
            for (Value result : call.getResults())
              updateFact(result, Fact::scalar(), facts, worklist, queued,
                         reached);
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
            if (!exact || !handles.count(*exact)) {
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
            Fact state = joinFacts(handles[*exact], Fact::nullHandle());
            state = joinFacts(state, resource);
            updateHandle(*exact, state, handles, worklist, queued, reached);
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
            if (!exact || !handles.count(*exact)) {
              call.emitError("udonclDeviceBufferGetRaw requires an exact "
                             "8-byte handle object");
              invalid = true;
              continue;
            }
            Fact state = handles[*exact];
            if (!state.initialized)
              state = Fact::nullHandle();
            recordCase(operandCases, call, invocation, 0, state);
            Value cell = call.getOperand(1);
            if (auto global = globalCell(cell)) {
              StringRef name = global.getSymName();
              resourceGlobals.insert(name);
              globalWrites[name] = joinFacts(globalWrites[name], state);
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
            if (!exact || !handles.count(*exact)) {
              call.emitError("udonclDeviceBufferDestroy requires an exact "
                             "8-byte handle object: ")
                  << call.getOperand(0);
              invalid = true;
              continue;
            }
            Fact state = handles[*exact];
            if (!state.initialized)
              state = Fact::nullHandle();
            recordCase(operandCases, call, invocation, 0, state);
            updateHandle(*exact, Fact::nullHandle(), handles, worklist, queued,
                         reached);
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
            if (!exactHandle || !handles.count(*exactHandle)) {
              call.emitError("udonclDeviceBufferView requires an exact "
                             "8-byte handle object");
              invalid = true;
              continue;
            }
            if (!exactOutput || !canHoldView(*exactOutput)) {
              call.emitError("udonclDeviceBufferView requires an exact "
                             "20-byte view subobject");
              invalid = true;
              continue;
            }
            Fact resource = handles[*exactHandle];
            ViewState previous = localViews[*exactOutput];
            ViewState next;
            next.hasPath = true;
            next.initialized = true;
            next.mayBeInvalid = true;
            next.resource = previous.initialized
                                ? joinFacts(previous.resource, resource)
                                : resource;
            localViews[*exactOutput] = next;
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
              call.emitError("udononnxCopyBufferView requires an initialized "
                             "source view subobject");
              invalid = true;
              continue;
            }
            if (!exactDestination || !canHoldView(*exactDestination)) {
              call.emitError("udononnxCopyBufferView requires an exact "
                             "destination view subobject: ")
                  << call.getOperand(0);
              invalid = true;
              continue;
            }
            localViews[*exactDestination] = localViews[*exactSource];
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
            auto exactView = view == objectValues.end()
                                 ? std::nullopt
                                 : exactObject(view->second);
            if (!exactView || !canHoldView(*exactView) ||
                !localViews[*exactView].initialized ||
                !localViews[*exactView].resource.resource ||
                localViews[*exactView].resource.unknown) {
              call.emitError("udonclBufferViewResolve requires an initialized "
                             "view subobject");
              invalid = true;
              continue;
            }
            Fact resource = localViews[*exactView].resource;
            recordCase(operandCases, call, invocation, 0, resource);
            Value cell = call.getOperand(1);
            if (auto global = globalCell(cell)) {
              StringRef name = global.getSymName();
              resourceGlobals.insert(name);
              globalWrites[name] = joinFacts(globalWrites[name], resource);
            } else if (eligibleCells.contains(cell)) {
              CellState previous = localMemory.lookup(cell);
              Fact output = previous.initializedOnAllPaths
                                ? joinFacts(previous.fact, resource)
                                : resource;
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
                        localPointerCells);
            handles = std::move(nestedResult.handles);
            localViews = std::move(nestedResult.views);
            localPointerCells = std::move(nestedResult.pointerCells);
            if (call.getNumResults() == 1 &&
                (!nestedResult.pointerExitCases.exact.empty() ||
                 nestedResult.pointerExitCases.unknown))
              pendingPointerEffects[call.getResult(0)] = {
                  call, std::move(nestedResult.pointerExitCases)};
            for (auto [index, result] : llvm::enumerate(call.getResults()))
              updateFact(result,
                         index < nestedResult.results.size()
                             ? nestedResult.results[index]
                             : Fact::scalar(),
                         facts, worklist, queued, reached);
            for (auto [index, result] : llvm::enumerate(call.getResults()))
              if (index < nestedResult.integerResults.size() &&
                  nestedResult.integerResults[index])
                integerValues[result] = *nestedResult.integerResults[index];
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
                  "resource use has unresolved non-handle provenance");
              invalid = true;
            }
            if (hasResource && !isKnownConsumer(calleeName)) {
              call.emitError(
                  "static device resource escapes to an unknown external call");
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
                     address.zero ? Fact::zeroScalar() : Fact::scalar(), facts,
                     worklist, queued, reached);
        } else if (auto pointerToInt =
                       dyn_cast<::vrc::udonhl::PtrToIntOp>(operation)) {
          updateFact(pointerToInt.getResult(), Fact::scalar(), facts, worklist,
                     queued, reached);
        } else if (auto pointerAdd =
                       dyn_cast<::vrc::udonhl::PtrAddOp>(operation)) {
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
          }
          updateFact(pointerAdd.getResult(), Fact::scalar(), facts, worklist,
                     queued, reached);
        } else if (auto add = dyn_cast<arith::AddIOp>(operation)) {
          if (valueFact(add.getLhs(), facts).resource ||
              valueFact(add.getRhs(), facts).resource) {
            add.emitError(
                "unsupported operation consumes a static device resource");
            invalid = true;
          }
          auto left = integerValues.find(add.getLhs());
          auto right = integerValues.find(add.getRhs());
          if (left != integerValues.end() && right != integerValues.end()) {
            auto folded = foldExactInteger(
                add.getType(), left->second, right->second,
                [](const llvm::APInt &lhs, const llvm::APInt &rhs) {
                  return lhs + rhs;
                });
            if (folded)
              integerValues[add.getResult()] = *folded;
          }
          updateFact(add.getResult(), Fact::scalar(), facts, worklist, queued,
                     reached);
        } else if (auto subtract = dyn_cast<arith::SubIOp>(operation)) {
          if (valueFact(subtract.getLhs(), facts).resource ||
              valueFact(subtract.getRhs(), facts).resource) {
            subtract.emitError(
                "unsupported operation consumes a static device resource");
            invalid = true;
          }
          auto left = integerValues.find(subtract.getLhs());
          auto right = integerValues.find(subtract.getRhs());
          if (left != integerValues.end() && right != integerValues.end()) {
            auto folded = foldExactInteger(
                subtract.getType(), left->second, right->second,
                [](const llvm::APInt &lhs, const llvm::APInt &rhs) {
                  return lhs - rhs;
                });
            if (folded)
              integerValues[subtract.getResult()] = *folded;
          }
          updateFact(subtract.getResult(), Fact::scalar(), facts, worklist,
                     queued, reached);
        } else if (auto multiply = dyn_cast<arith::MulIOp>(operation)) {
          if (valueFact(multiply.getLhs(), facts).resource ||
              valueFact(multiply.getRhs(), facts).resource) {
            multiply.emitError(
                "unsupported operation consumes a static device resource");
            invalid = true;
          }
          auto left = integerValues.find(multiply.getLhs());
          auto right = integerValues.find(multiply.getRhs());
          if (left != integerValues.end() && right != integerValues.end()) {
            auto folded = foldExactInteger(
                multiply.getType(), left->second, right->second,
                [](const llvm::APInt &lhs, const llvm::APInt &rhs) {
                  return lhs * rhs;
                });
            if (folded)
              integerValues[multiply.getResult()] = *folded;
          }
          updateFact(multiply.getResult(), Fact::scalar(), facts, worklist,
                     queued, reached);
        } else if (auto compare = dyn_cast<arith::CmpIOp>(operation)) {
          if (valueFact(compare.getLhs(), facts).resource ||
              valueFact(compare.getRhs(), facts).resource) {
            compare.emitError(
                "unsupported operation consumes a static device resource");
            invalid = true;
          }
          if (compare.getPredicate() == arith::CmpIPredicate::eq ||
              compare.getPredicate() == arith::CmpIPredicate::ne) {
            Value subject;
            std::optional<int64_t> constant;
            if (pendingPointerEffects.contains(compare.getLhs())) {
              subject = compare.getLhs();
              auto found = integerValues.find(compare.getRhs());
              if (found != integerValues.end())
                constant = found->second;
            } else if (pendingPointerEffects.contains(compare.getRhs())) {
              subject = compare.getRhs();
              auto found = integerValues.find(compare.getLhs());
              if (found != integerValues.end())
                constant = found->second;
            }
            if (subject && constant)
              comparisonRefinements[compare.getResult()] = {
                  subject, *constant,
                  compare.getPredicate() == arith::CmpIPredicate::eq};
          }
          updateFact(compare.getResult(), Fact::scalar(), facts, worklist,
                     queued, reached);
        } else if (auto select = dyn_cast<arith::SelectOp>(operation)) {
          auto trueObject = objectValues.find(select.getTrueValue());
          auto falseObject = objectValues.find(select.getFalseValue());
          if (trueObject != objectValues.end() ||
              falseObject != objectValues.end()) {
            ObjectRefSet selected;
            if (trueObject != objectValues.end())
              selected = trueObject->second;
            else
              selected.unknown = true;
            if (falseObject != objectValues.end())
              selected =
                  joinObjectRefs(std::move(selected), falseObject->second);
            else
              selected.unknown = true;
            objectValues[select.getResult()] = std::move(selected);
          }
          auto trueInteger = integerValues.find(select.getTrueValue());
          auto falseInteger = integerValues.find(select.getFalseValue());
          if (trueInteger != integerValues.end() &&
              falseInteger != integerValues.end() &&
              trueInteger->second == falseInteger->second)
            integerValues[select.getResult()] = trueInteger->second;
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
          mergeSuccessor(branch.getDest(), branch.getDestOperands(), facts,
                         worklist, queued, reached, block, localMemory,
                         memoryInputs, edgeMemory, eligibleCells, objectValues,
                         integerValues, ambiguousIntegers, handleFieldValues,
                         localViews, viewInputs, edgeViews, localPointerCells,
                         pointerInputs, edgePointers, edgeIntegerArguments);
        } else if (auto branch = dyn_cast<cf::CondBranchOp>(operation)) {
          PointerCellMap truePointers = localPointerCells;
          PointerCellMap falsePointers = localPointerCells;
          auto refinement = comparisonRefinements.find(branch.getCondition());
          if (refinement != comparisonRefinements.end()) {
            auto effect =
                pendingPointerEffects.find(refinement->second.subject);
            Operation *comparison = branch.getCondition().getDefiningOp();
            bool direct = effect != pendingPointerEffects.end() && comparison &&
                          effect->second.call->getBlock() == block &&
                          comparison->getBlock() == block;
            if (direct)
              for (Operation *between = effect->second.call->getNextNode();
                   between && between != &operation;
                   between = between->getNextNode())
                if (between != comparison) {
                  direct = false;
                  break;
                }
            if (direct) {
              truePointers = selectPointerExitCases(
                  effect->second.cases, refinement->second.constant,
                  refinement->second.equalOnTrue, localPointerCells);
              falsePointers = selectPointerExitCases(
                  effect->second.cases, refinement->second.constant,
                  !refinement->second.equalOnTrue, localPointerCells);
            }
          }
          mergeSuccessor(branch.getTrueDest(), branch.getTrueDestOperands(),
                         facts, worklist, queued, reached, block, localMemory,
                         memoryInputs, edgeMemory, eligibleCells, objectValues,
                         integerValues, ambiguousIntegers, handleFieldValues,
                         localViews, viewInputs, edgeViews, truePointers,
                         pointerInputs, edgePointers, edgeIntegerArguments);
          mergeSuccessor(branch.getFalseDest(), branch.getFalseDestOperands(),
                         facts, worklist, queued, reached, block, localMemory,
                         memoryInputs, edgeMemory, eligibleCells, objectValues,
                         integerValues, ambiguousIntegers, handleFieldValues,
                         localViews, viewInputs, edgeViews, falsePointers,
                         pointerInputs, edgePointers, edgeIntegerArguments);
        } else if (auto branch = dyn_cast<cf::SwitchOp>(operation)) {
          mergeSuccessor(branch.getDefaultDestination(),
                         branch.getDefaultOperands(), facts, worklist, queued,
                         reached, block, localMemory, memoryInputs, edgeMemory,
                         eligibleCells, objectValues, integerValues,
                         ambiguousIntegers, handleFieldValues, localViews,
                         viewInputs, edgeViews, localPointerCells,
                         pointerInputs, edgePointers, edgeIntegerArguments);
          for (auto [index, destination] :
               llvm::enumerate(branch.getCaseDestinations()))
            mergeSuccessor(destination, branch.getCaseOperands(index), facts,
                           worklist, queued, reached, block, localMemory,
                           memoryInputs, edgeMemory, eligibleCells,
                           objectValues, integerValues, ambiguousIntegers,
                           handleFieldValues, localViews, viewInputs, edgeViews,
                           localPointerCells, pointerInputs, edgePointers,
                           edgeIntegerArguments);
        } else if (auto returned = dyn_cast<func::ReturnOp>(operation)) {
          bool exported = function->hasAttr("udoncl.export") ||
                          function->hasAttr("vrc.export_id");
          for (auto [index, operand] :
               llvm::enumerate(returned.getOperands())) {
            Fact fact = valueFact(operand, facts);
            if (exported && fact.resource) {
              returned.emitError(
                  "static device resource escapes through an exported result");
              invalid = true;
            }
            returns[index] = joinFacts(returns[index], fact);
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
            returnedViews = localViews;
            sawReturn = true;
          } else {
            returnedPointerCells =
                joinPointerMaps(returnedPointerCells, localPointerCells);
            returnedViews = joinViewMaps(returnedViews, localViews);
          }
          if (returned.getNumOperands() == 1 &&
              function.getFunctionType().getNumResults() == 1 &&
              function.getFunctionType().getResult(0).isIntOrIndex()) {
            Value status = returned.getOperand(0);
            auto pending = pendingPointerEffects.find(status);
            if (pending != pendingPointerEffects.end() &&
                pending->second.call->getBlock() == block &&
                pending->second.call->getNextNode() == &operation) {
              for (const auto &[value, pointerCells] :
                   pending->second.cases.exact)
                addPointerExitCase(pointerExitCases, value, pointerCells);
              if (pending->second.cases.unknown)
                addPointerExitCase(pointerExitCases, std::nullopt,
                                   *pending->second.cases.unknown);
            } else if (auto integer = integerValues.find(status);
                       integer != integerValues.end()) {
              addPointerExitCase(pointerExitCases, integer->second,
                                 localPointerCells);
            } else if (auto argument = dyn_cast<BlockArgument>(status);
                       argument && argument.getOwner() == block &&
                       llvm::hasSingleElement(*block)) {
              for (Block *predecessor : block->getPredecessors()) {
                auto integers = edgeIntegerArguments.find({predecessor, block});
                auto pointers = edgePointers.find({predecessor, block});
                if (pointers == edgePointers.end())
                  continue;
                std::optional<int64_t> value;
                if (integers != edgeIntegerArguments.end() &&
                    argument.getArgNumber() < integers->second.size())
                  value = integers->second[argument.getArgNumber()];
                addPointerExitCase(pointerExitCases, value, pointers->second);
              }
            } else {
              addPointerExitCase(pointerExitCases, std::nullopt,
                                 localPointerCells);
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
                << operation;
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
    active.erase(function);
    for (auto [index, fact] : llvm::enumerate(returns))
      recordCase(functionResultCases, function, invocation, index, fact);
    return {std::move(returns),
            std::move(integerReturns),
            std::move(handleFieldReturns),
            std::move(handles),
            std::move(returnedViews),
            std::move(returnedPointerCells),
            std::move(pointerExitCases)};
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
      auto incomingObject = objectValues.find(operand);
      if (incomingObject != objectValues.end()) {
        auto existingObject = objectValues.find(argument);
        if (existingObject == objectValues.end())
          objectValues[argument] = incomingObject->second;
        else
          existingObject->second = joinObjectRefs(
              std::move(existingObject->second), incomingObject->second);
      } else {
        objectValues[argument].unknown = true;
      }
      auto incomingInteger = integerValues.find(operand);
      incomingIntegers.push_back(
          incomingInteger == integerValues.end()
              ? std::nullopt
              : std::optional<int64_t>(incomingInteger->second));
      if (!ambiguousIntegers.contains(argument)) {
        auto existingInteger = integerValues.find(argument);
        if (incomingInteger == integerValues.end()) {
          integerValues.erase(argument);
          ambiguousIntegers.insert(argument);
        } else if (existingInteger == integerValues.end()) {
          integerValues[argument] = incomingInteger->second;
        } else if (existingInteger->second != incomingInteger->second) {
          integerValues.erase(argument);
          ambiguousIntegers.insert(argument);
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
    edgeViews[{source, destination}] = sourceViews;
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
    edgePointers[{source, destination}] = sourcePointers;
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
        recomputedPointers = joinPointerMaps(recomputedPointers, edge->second);
      }
    }
    if (pointerInputs[destination] != recomputedPointers)
      changed = true;
    pointerInputs[destination] = std::move(recomputedPointers);
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
    globalHandleFacts.clear();
    globalHandleWrites.clear();
    localHandleObjects.clear();
    objectSizes.clear();
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
    for (func::FuncOp function : module.getOps<func::FuncOp>())
      functions.try_emplace(function.getSymName(), function);
    DataLayout layout(module);
    for (auto global : module.getOps<::vrc::udonhl::GlobalOp>()) {
      auto initial =
          dyn_cast_or_null<IntegerAttr>(global.getInitialValueAttr());
      auto pointer = dyn_cast<::vrc::udonhl::PtrType>(global.getStorageType());
      if (global.getLinkage() == "internal" && !global.getConstant() &&
          global.getAddressSpace() == 0 && global.getAlignment() == 4 &&
          pointer && pointer.getAddressSpace() == 0 && initial &&
          initial.getInt() == 0)
        eligibleGlobals.try_emplace(global.getSymName(), global);

      auto bytes =
          dyn_cast_or_null<DenseI8ArrayAttr>(global.getInitialValueAttr());
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
        objectSizes[object] = static_cast<uint32_t>(handleBytes);
        for (uint32_t offset = 0; offset < handleBytes; offset += 8)
          globalHandleFacts[{object, static_cast<int32_t>(offset)}] =
              Fact::nullHandle();
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
      if (!function.isExternal() && (function->hasAttr("udoncl.export") ||
                                     function->hasAttr("vrc.export_id")))
        roots.push_back(function);
    llvm::sort(roots, [](func::FuncOp left, func::FuncOp right) {
      return left.getSymName() < right.getSymName();
    });
    bool converged = false;
    for (unsigned iteration = 0; iteration != 100 && !invalid; ++iteration) {
      globalWrites.clear();
      globalHandleWrites.clear();
      for (auto &[name, global] : eligibleGlobals)
        globalWrites[name] = Fact();
      for (func::FuncOp root : roots) {
        SmallVector<Fact> arguments(root.getNumArguments(), Fact::scalar());
        SmallVector<std::optional<int64_t>> argumentIntegers(
            root.getNumArguments(), std::nullopt);
        SmallVector<std::optional<ObjectRefSet>> argumentObjects(
            root.getNumArguments(), std::nullopt);
        SmallVector<HandleFieldOriginSet> argumentHandleFields(
            root.getNumArguments());
        HandleMap handles;
        ViewMap views;
        PointerCellMap pointerCells;
        handles = globalHandleFacts;
        AnalysisResult result = analyze(
            root,
            {FlatSymbolRefAttr::get(module.getContext(), root.getSymName()),
             builder->getArrayAttr({})},
            arguments, argumentIntegers, argumentObjects, argumentHandleFields,
            handles, views, pointerCells);
        for (auto &[name, global] : eligibleHandleGlobals) {
          ObjectId object = globalHandleObjects[name];
          uint32_t bytes = objectSizes[object];
          for (uint32_t offset = 0; offset < bytes; offset += 8) {
            ObjectRef handle{object, static_cast<int32_t>(offset)};
            globalHandleWrites[handle] =
                joinFacts(globalHandleWrites[handle], result.handles[handle]);
          }
        }
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
      for (auto &[handle, fact] : globalHandleFacts) {
        Fact next = joinFacts(Fact::nullHandle(), globalHandleWrites[handle]);
        if (!(fact == next)) {
          fact = next;
          converged = false;
        }
      }
      if (converged)
        break;
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
