#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/STLExtras.h"
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

struct CellState {
  bool hasPath = false;
  bool initializedOnAllPaths = false;
  Fact fact;
};

using CellMap = llvm::DenseMap<Value, CellState>;
using ObjectId = uint32_t;
using HandleMap = llvm::DenseMap<ObjectId, Fact>;

struct ObjectRef {
  ObjectId base = 0;
  int32_t offset = 0;

  bool operator<(const ObjectRef &other) const {
    return std::tie(base, offset) < std::tie(other.base, other.offset);
  }
};

struct ViewState {
  bool initialized = false;
  bool mayBeInvalid = false;
  Fact resource;
};

using ViewMap = std::map<ObjectRef, ViewState>;

struct AnalysisResult {
  SmallVector<Fact> results;
  HandleMap handles;
  ViewMap views;
};

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
  llvm::StringMap<ObjectId> globalHandleObjects;
  llvm::StringMap<Fact> globalHandleFacts;
  llvm::StringMap<Fact> globalHandleWrites;
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

  void updateHandle(ObjectId object, const Fact &incoming, HandleMap &handles,
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
                         ArrayRef<std::optional<ObjectRef>> argumentObjects,
                         const HandleMap &incomingHandles,
                         const ViewMap &incomingViews) {
    if (!active.insert(function).second) {
      if (llvm::none_of(arguments,
                        [](const Fact &fact) { return fact.resource; }))
        return {SmallVector<Fact>(function.getFunctionType().getNumResults(),
                                  Fact::scalar()),
                incomingHandles, incomingViews};
      function.emitError(
          "resource provenance through recursion is unsupported");
      invalid = true;
      return {SmallVector<Fact>(), HandleMap(), ViewMap()};
    }
    llvm::DenseMap<Value, Fact> facts;
    llvm::DenseMap<Value, ObjectRef> objectValues;
    HandleMap handles = incomingHandles;
    ViewMap views = incomingViews;
    llvm::DenseSet<Value> eligibleCells;
    function.walk([&](::vrc::udonhl::AllocaOp allocation) {
      if (isSingletonCell(allocation))
        eligibleCells.insert(allocation.getResult());
      auto bytes = constantAllocationBytes(allocation);
      if (bytes && (*bytes == 8 || *bytes >= 20)) {
        ObjectId object = localObject(invocation, allocation, *bytes);
        objectValues[allocation.getResult()] = {object, 0};
        if (*bytes == 8)
          handles.try_emplace(object, Fact::nullHandle());
      }
    });
    function.walk([&](::vrc::udonhl::AddressOfOp address) {
      auto found = globalHandleObjects.find(address.getGlobalAttr().getValue());
      if (found != globalHandleObjects.end())
        objectValues[address.getResult()] = {found->getValue(), 0};
    });
    for (auto [index, argument] : llvm::enumerate(function.getArguments())) {
      Fact fact = index < arguments.size() ? arguments[index] : Fact::scalar();
      facts[argument] = fact;
      if (index < argumentObjects.size() && argumentObjects[index])
        objectValues[argument] = *argumentObjects[index];
      recordCase(argumentCases, function, invocation, index, fact);
    }

    SmallVector<Fact> returns(function.getFunctionType().getNumResults());
    SmallVector<Block *> worklist;
    llvm::DenseSet<Block *> queued;
    llvm::DenseSet<Block *> reached;
    llvm::DenseMap<Block *, CellMap> memoryInputs;
    llvm::DenseMap<std::pair<Block *, Block *>, CellMap> edgeMemory;
    llvm::DenseSet<Value> resourceCells;
    if (!function.empty()) {
      worklist.push_back(&function.front());
      queued.insert(&function.front());
      reached.insert(&function.front());
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
      for (Operation &operation : *block) {
        for (auto [index, operand] : llvm::enumerate(operation.getOperands()))
          recordCase(operandCases, &operation, invocation, index,
                     valueFact(operand, facts));

        if (auto store = dyn_cast<::vrc::udonhl::StoreOp>(operation)) {
          Fact stored = valueFact(store.getValue(), facts);
          Value cell = store.getPointer();
          Fact pointer = valueFact(cell, facts);
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
            if (stored.resource) {
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
            if (object == objectValues.end() || object->second.offset != 0 ||
                !handles.contains(object->second.base)) {
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
            Fact state =
                joinFacts(handles[object->second.base], Fact::nullHandle());
            state = joinFacts(state, resource);
            updateHandle(object->second.base, state, handles, worklist, queued,
                         reached);
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
            if (object == objectValues.end() || object->second.offset != 0 ||
                !handles.contains(object->second.base)) {
              call.emitError("udonclDeviceBufferGetRaw requires an exact "
                             "8-byte handle object");
              invalid = true;
              continue;
            }
            Fact state = handles[object->second.base];
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
            if (object == objectValues.end() || object->second.offset != 0 ||
                !handles.contains(object->second.base)) {
              call.emitError("udonclDeviceBufferDestroy requires an exact "
                             "8-byte handle object");
              invalid = true;
              continue;
            }
            Fact state = handles[object->second.base];
            if (!state.initialized)
              state = Fact::nullHandle();
            recordCase(operandCases, call, invocation, 0, state);
            updateHandle(object->second.base, Fact::nullHandle(), handles,
                         worklist, queued, reached);
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
            if (handle == objectValues.end() || handle->second.offset != 0 ||
                !handles.contains(handle->second.base)) {
              call.emitError("udonclDeviceBufferView requires an exact "
                             "8-byte handle object");
              invalid = true;
              continue;
            }
            if (output == objectValues.end() || !canHoldView(output->second)) {
              call.emitError("udonclDeviceBufferView requires an exact "
                             "20-byte view subobject");
              invalid = true;
              continue;
            }
            Fact resource = handles[handle->second.base];
            ViewState previous = views[output->second];
            ViewState next;
            next.initialized = true;
            next.mayBeInvalid = true;
            next.resource = previous.initialized
                                ? joinFacts(previous.resource, resource)
                                : resource;
            views[output->second] = next;
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
            if (source == objectValues.end() || !canHoldView(source->second) ||
                !views[source->second].initialized) {
              call.emitError("udononnxCopyBufferView requires an initialized "
                             "source view subobject");
              invalid = true;
              continue;
            }
            if (destination == objectValues.end() ||
                !canHoldView(destination->second)) {
              call.emitError("udononnxCopyBufferView requires an exact "
                             "destination view subobject");
              invalid = true;
              continue;
            }
            views[destination->second] = views[source->second];
            recordCase(operandCases, call, invocation, 0,
                       views[destination->second].resource);
            recordCase(operandCases, call, invocation, 1,
                       views[source->second].resource);
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
            if (view == objectValues.end() || !canHoldView(view->second) ||
                !views[view->second].initialized) {
              call.emitError("udonclBufferViewResolve requires an initialized "
                             "view subobject");
              invalid = true;
              continue;
            }
            Fact resource = views[view->second].resource;
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
            SmallVector<std::optional<ObjectRef>> actualObjects;
            for (Value operand : call.getOperands()) {
              actuals.push_back(valueFact(operand, facts));
              auto object = objectValues.find(operand);
              actualObjects.push_back(
                  object == objectValues.end()
                      ? std::nullopt
                      : std::optional<ObjectRef>(object->second));
            }
            Invocation nested = calleeInvocation(invocation, function, call);
            AnalysisResult nestedResult =
                analyze(found->getValue(), nested, actuals, actualObjects,
                        handles, views);
            handles = std::move(nestedResult.handles);
            views = std::move(nestedResult.views);
            for (auto [index, result] : llvm::enumerate(call.getResults()))
              updateFact(result,
                         index < nestedResult.results.size()
                             ? nestedResult.results[index]
                             : Fact::scalar(),
                         facts, worklist, queued, reached);
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
        } else if (auto pointerAdd =
                       dyn_cast<::vrc::udonhl::PtrAddOp>(operation)) {
          auto base = objectValues.find(pointerAdd.getPointer());
          auto offset =
              pointerAdd.getByteOffset().getDefiningOp<arith::ConstantIntOp>();
          if (base != objectValues.end() && offset) {
            int64_t combined =
                static_cast<int64_t>(base->second.offset) + offset.value();
            if (combined >= std::numeric_limits<int32_t>::min() &&
                combined <= std::numeric_limits<int32_t>::max())
              objectValues[pointerAdd.getResult()] = {
                  base->second.base, static_cast<int32_t>(combined)};
          }
          updateFact(pointerAdd.getResult(), Fact::scalar(), facts, worklist,
                     queued, reached);
        } else if (auto select = dyn_cast<arith::SelectOp>(operation)) {
          updateFact(select.getResult(),
                     joinFacts(valueFact(select.getTrueValue(), facts),
                               valueFact(select.getFalseValue(), facts)),
                     facts, worklist, queued, reached);
        } else if (auto branch = dyn_cast<cf::BranchOp>(operation)) {
          mergeSuccessor(branch.getDest(), branch.getDestOperands(), facts,
                         worklist, queued, reached, block, localMemory,
                         memoryInputs, edgeMemory, eligibleCells);
        } else if (auto branch = dyn_cast<cf::CondBranchOp>(operation)) {
          mergeSuccessor(branch.getTrueDest(), branch.getTrueDestOperands(),
                         facts, worklist, queued, reached, block, localMemory,
                         memoryInputs, edgeMemory, eligibleCells);
          mergeSuccessor(branch.getFalseDest(), branch.getFalseDestOperands(),
                         facts, worklist, queued, reached, block, localMemory,
                         memoryInputs, edgeMemory, eligibleCells);
        } else if (auto branch = dyn_cast<cf::SwitchOp>(operation)) {
          mergeSuccessor(branch.getDefaultDestination(),
                         branch.getDefaultOperands(), facts, worklist, queued,
                         reached, block, localMemory, memoryInputs, edgeMemory,
                         eligibleCells);
          for (auto [index, destination] :
               llvm::enumerate(branch.getCaseDestinations()))
            mergeSuccessor(destination, branch.getCaseOperands(index), facts,
                           worklist, queued, reached, block, localMemory,
                           memoryInputs, edgeMemory, eligibleCells);
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
                "unsupported operation consumes a static device resource");
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
    return {std::move(returns), std::move(handles), std::move(views)};
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
      const llvm::DenseSet<Value> &eligibleCells) {
    bool changed = false;
    for (auto [argument, operand] :
         llvm::zip(destination->getArguments(), operands)) {
      if (updateFact(argument, valueFact(operand, facts), facts, worklist,
                     queued, reached)) {
        changed = true;
      }
    }
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
      if (global.getLinkage() == "internal" && !global.getConstant() &&
          global.getAddressSpace() == 0 && global.getAlignment() >= 4 &&
          array && array.getNumElements() == 8 &&
          array.getElementType().isInteger(8) && bytes && bytes.size() == 8 &&
          llvm::all_of(bytes.asArrayRef(),
                       [](int8_t byte) { return byte == 0; })) {
        eligibleHandleGlobals.try_emplace(global.getSymName(), global);
        globalHandleObjects[global.getSymName()] = nextObjectId++;
        objectSizes[globalHandleObjects[global.getSymName()]] = 8;
        globalHandleFacts[global.getSymName()] = Fact::nullHandle();
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
      for (auto &[name, global] : eligibleHandleGlobals)
        globalHandleWrites[name] = Fact();
      for (func::FuncOp root : roots) {
        SmallVector<Fact> arguments(root.getNumArguments(), Fact::scalar());
        SmallVector<std::optional<ObjectRef>> argumentObjects(
            root.getNumArguments(), std::nullopt);
        HandleMap handles;
        ViewMap views;
        for (auto &[name, global] : eligibleHandleGlobals)
          handles[globalHandleObjects[name]] = globalHandleFacts[name];
        AnalysisResult result = analyze(
            root,
            {FlatSymbolRefAttr::get(module.getContext(), root.getSymName()),
             builder->getArrayAttr({})},
            arguments, argumentObjects, handles, views);
        for (auto &[name, global] : eligibleHandleGlobals) {
          ObjectId object = globalHandleObjects[name];
          globalHandleWrites[name] =
              joinFacts(globalHandleWrites[name], result.handles[object]);
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
      for (auto &[name, global] : eligibleHandleGlobals) {
        Fact next = joinFacts(Fact::nullHandle(), globalHandleWrites[name]);
        if (!(globalHandleFacts[name] == next)) {
          globalHandleFacts[name] = next;
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
