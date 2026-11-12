#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

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
           return isExactCellAccess(use, allocation.getResult());
         });
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
  bool invalid = false;

  Fact valueFact(Value value, llvm::DenseMap<Value, Fact> &facts) {
    auto found = facts.find(value);
    if (found != facts.end())
      return found->second;
    return Fact::scalar();
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

  SmallVector<Fact> analyze(func::FuncOp function, const Invocation &invocation,
                            ArrayRef<Fact> arguments) {
    if (!active.insert(function).second) {
      if (llvm::none_of(arguments,
                        [](const Fact &fact) { return fact.resource; }))
        return SmallVector<Fact>(function.getFunctionType().getNumResults(),
                                 Fact::scalar());
      function.emitError(
          "resource provenance through recursion is unsupported");
      invalid = true;
      return {};
    }
    llvm::DenseMap<Value, Fact> facts;
    llvm::DenseSet<Value> eligibleCells;
    function.walk([&](::vrc::udonhl::AllocaOp allocation) {
      if (isSingletonCell(allocation))
        eligibleCells.insert(allocation.getResult());
    });
    for (auto [index, argument] : llvm::enumerate(function.getArguments())) {
      Fact fact = index < arguments.size() ? arguments[index] : Fact::scalar();
      facts[argument] = fact;
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
          if (eligibleCells.contains(cell)) {
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
          if (eligibleCells.contains(cell)) {
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
          if (isAllocation(calleeName)) {
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
          } else if (auto found = functions.find(calleeName);
                     found != functions.end() &&
                     !found->getValue().isExternal()) {
            SmallVector<Fact> actuals;
            for (Value operand : call.getOperands())
              actuals.push_back(valueFact(operand, facts));
            Invocation nested = calleeInvocation(invocation, function, call);
            SmallVector<Fact> results =
                analyze(found->getValue(), nested, actuals);
            for (auto [index, result] : llvm::enumerate(call.getResults()))
              updateFact(result,
                         index < results.size() ? results[index]
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
    return returns;
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

  std::string invocationPathKey(ArrayAttr path) {
    std::string storage;
    llvm::raw_string_ostream stream(storage);
    path.print(stream);
    return storage;
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
    for (func::FuncOp function : module.getOps<func::FuncOp>())
      functions.try_emplace(function.getSymName(), function);
    SmallVector<func::FuncOp> roots;
    for (func::FuncOp function : module.getOps<func::FuncOp>())
      if (!function.isExternal() && (function->hasAttr("udoncl.export") ||
                                     function->hasAttr("vrc.export_id")))
        roots.push_back(function);
    llvm::sort(roots, [](func::FuncOp left, func::FuncOp right) {
      return left.getSymName() < right.getSymName();
    });
    for (func::FuncOp root : roots) {
      SmallVector<Fact> arguments(root.getNumArguments(), Fact::scalar());
      analyze(root,
              {FlatSymbolRefAttr::get(module.getContext(), root.getSymName()),
               builder->getArrayAttr({})},
              arguments);
      if (invalid)
        break;
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
