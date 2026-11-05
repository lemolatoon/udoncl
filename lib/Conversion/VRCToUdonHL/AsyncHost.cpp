#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/VRC/VRCDialect.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <set>
#include <string>

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_LOWERHOSTTOASYNCPASS
#define GEN_PASS_DEF_INSERTUDONCOOPERATIVECHECKPOINTSPASS
#define GEN_PASS_DEF_LOWERHOSTASYNCTONATIVEPASS
#define GEN_PASS_DEF_PREPAREUDONASYNCRUNTIMEPASS
#define GEN_PASS_DEF_LOWERUDONASYNCRUNTIMEPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {
namespace runtime = ::vrc::vrc;

struct InsertUdonCooperativeCheckpointsPass
    : impl::InsertUdonCooperativeCheckpointsPassBase<
          InsertUdonCooperativeCheckpointsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    constexpr unsigned defaultMinimumOperations = 256;
    // Raw-memory Host operations around a device dispatch expand to many Udon
    // instructions.  Counting twenty MLIR operations lets short-looking
    // wrappers exceed the VM's per-event watchdog before they receive a
    // cooperative checkpoint.
    constexpr unsigned transferMinimumOperations = 12;
    llvm::DenseSet<Operation *> deviceDispatchFunctions;
    for (func::FuncOp function : module.getOps<func::FuncOp>())
      function.walk([&](runtime::DispatchOp) {
        deviceDispatchFunctions.insert(function.getOperation());
      });
    bool changed = true;
    while (changed) {
      changed = false;
      for (func::FuncOp function : module.getOps<func::FuncOp>()) {
        if (deviceDispatchFunctions.contains(function.getOperation()))
          continue;
        bool reachesDispatch = false;
        function.walk([&](func::CallOp call) {
          auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee());
          reachesDispatch |=
              callee && deviceDispatchFunctions.contains(callee.getOperation());
        });
        if (reachesDispatch) {
          deviceDispatchFunctions.insert(function.getOperation());
          changed = true;
        }
      }
    }
    auto baseOperationCost = [](Operation &operation) -> uint64_t {
      if (isa<::vrc::udonhl::LoadOp, ::vrc::udonhl::StoreOp>(operation))
        return 16;
      if (isa<::vrc::udonhl::AddressOfOp, ::vrc::udonhl::AllocaOp>(operation))
        return 8;
      return 1;
    };
    llvm::DenseMap<Operation *, uint64_t> transitiveCosts;
    llvm::SmallPtrSet<Operation *, 32> costing;
    std::function<uint64_t(func::FuncOp)> functionCost =
        [&](func::FuncOp function) -> uint64_t {
      auto cached = transitiveCosts.find(function.getOperation());
      if (cached != transitiveCosts.end())
        return cached->second;
      if (!costing.insert(function.getOperation()).second)
        return 1;
      uint64_t cost = 0;
      for (Block &block : function.getBody())
        for (Operation &operation : block) {
          if (isa<arith::ConstantOp>(operation) ||
              operation.hasTrait<OpTrait::IsTerminator>())
            continue;
          uint64_t operationCost = baseOperationCost(operation);
          if (auto call = dyn_cast<func::CallOp>(operation)) {
            auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee());
            auto effect =
                callee ? callee->getAttrOfType<StringAttr>("vrc.async_effect")
                       : StringAttr();
            if (callee && (!effect || effect.getValue() != "may_suspend"))
              operationCost = functionCost(callee);
          }
          cost = std::min<uint64_t>(UINT32_MAX, cost + operationCost);
        }
      costing.erase(function.getOperation());
      transitiveCosts.try_emplace(function.getOperation(), cost);
      return cost;
    };
    for (func::FuncOp function : module.getOps<func::FuncOp>())
      functionCost(function);
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      if (function.isDeclaration())
        continue;
      auto effect = function->getAttrOfType<StringAttr>("vrc.async_effect");
      if (!effect || effect.getValue() != "may_suspend")
        continue;
      bool transfersBuffer = false;
      function.walk([&](Operation *operation) {
        transfersBuffer |=
            isa<runtime::BufferUploadOp, runtime::BufferReadbackOp,
                runtime::BufferCopyOp>(operation);
      });
      bool dispatchesDevice =
          deviceDispatchFunctions.contains(function.getOperation());
      unsigned minimumOperations = (transfersBuffer || dispatchesDevice)
                                       ? transferMinimumOperations
                                       : defaultMinimumOperations;
      SmallVector<std::pair<uint64_t, SmallVector<Operation *>>>
          checkpointRegions;
      SmallVector<Operation *> region;
      uint64_t regionCost = 0;
      auto operationCost = [&](Operation &operation) -> uint64_t {
        if (auto call = dyn_cast<func::CallOp>(operation)) {
          auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee());
          auto calleeEffect =
              callee ? callee->getAttrOfType<StringAttr>("vrc.async_effect")
                     : StringAttr();
          if (callee &&
              (!calleeEffect || calleeEffect.getValue() != "may_suspend"))
            return transitiveCosts.lookup(callee.getOperation());
        }
        return baseOperationCost(operation);
      };
      auto finishRegion = [&] {
        if (regionCost >= minimumOperations)
          checkpointRegions.emplace_back(regionCost, region);
        region.clear();
        regionCost = 0;
      };
      for (Block &block : function.getBody()) {
        for (Operation &operation : block) {
          bool suspensionBoundary = isa<runtime::AwaitOp>(operation);
          if (auto call = dyn_cast<func::CallOp>(operation)) {
            auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee());
            auto calleeEffect =
                callee ? callee->getAttrOfType<StringAttr>("vrc.async_effect")
                       : StringAttr();
            suspensionBoundary |=
                calleeEffect && calleeEffect.getValue() == "may_suspend";
          }
          if (suspensionBoundary) {
            finishRegion();
            continue;
          }
          if (isa<arith::ConstantOp>(operation) ||
              operation.hasTrait<OpTrait::IsTerminator>())
            continue;
          region.push_back(&operation);
          regionCost += operationCost(operation);
        }
      }
      finishRegion();
      if (checkpointRegions.empty())
        continue;
      llvm::sort(checkpointRegions, [](const auto &left, const auto &right) {
        return left.first > right.first;
      });
      for (const auto &[checkpointRegionCost, checkpointRegion] :
           ArrayRef(checkpointRegions)
               .take_front(std::min<size_t>(2, checkpointRegions.size()))) {
        llvm::SmallPtrSet<Operation *, 4> points;
        for (unsigned quarter = 1; quarter < 4; ++quarter) {
          uint64_t target =
              llvm::divideCeil(checkpointRegionCost * quarter, uint64_t{4});
          uint64_t prefixCost = 0;
          Operation *point = checkpointRegion.front();
          for (Operation *candidate : checkpointRegion) {
            point = candidate;
            prefixCost += operationCost(*candidate);
            if (prefixCost >= target)
              break;
          }
          if (!points.insert(point).second)
            continue;
          OpBuilder builder(point);
          auto token =
              arith::ConstantIntOp::create(builder, point->getLoc(), 0, 32);
          auto event = UnrealizedConversionCastOp::create(
              builder, point->getLoc(),
              TypeRange{runtime::EventType::get(module.getContext())},
              ValueRange{token});
          auto checkpoint = runtime::AwaitOp::create(builder, point->getLoc(),
                                                     builder.getI32Type(),
                                                     event.getResult(0));
          checkpoint->setAttr("vrc.cooperative_checkpoint",
                              builder.getUnitAttr());
        }
      }
    }
  }
};

bool isAsyncObjectType(Type type) {
  return isa<async::ValueType, async::TokenType>(type);
}

bool isRecomputableAcrossSuspend(Value value) {
  auto result = dyn_cast<OpResult>(value);
  if (!result)
    return false;
  StringRef name = result.getOwner()->getName().getStringRef();
  return name == "arith.constant" || name == "udonhl.address_of";
}

bool isCoroutineBookkeeping(Value value) {
  auto result = dyn_cast<OpResult>(value);
  return result &&
         result.getOwner()->getName().getStringRef().starts_with("async.coro.");
}

bool isSupportedSpillType(Type type) {
  if (auto integer = dyn_cast<IntegerType>(type))
    return integer.getWidth() <= 64;
  return type.isF32() ||
         isa<::vrc::udonhl::PtrType, runtime::BufferType, runtime::EventType,
             runtime::DispatchDescType>(type);
}

std::string typeName(Type type) {
  std::string result;
  llvm::raw_string_ostream output(result);
  type.print(output);
  return result;
}

Type physicalFrameType(Type type, MLIRContext *context) {
  if (isa<runtime::BufferType, runtime::EventType, runtime::DispatchDescType>(
          type))
    return IntegerType::get(context, 32);
  return type;
}

Attribute zeroFrameValue(Type type, Builder &builder) {
  if (auto integer = dyn_cast<IntegerType>(type))
    return builder.getIntegerAttr(integer, 0);
  if (type.isF32())
    return builder.getF32FloatAttr(0.0f);
  if (isa<::vrc::udonhl::PtrType>(type))
    return builder.getI32IntegerAttr(0);
  return {};
}

uint64_t frameAlignment(Type type) {
  if (auto integer = dyn_cast<IntegerType>(type))
    return integer.getWidth() <= 8    ? 1
           : integer.getWidth() <= 16 ? 2
           : integer.getWidth() <= 32 ? 4
                                      : 8;
  return 4;
}

bool setsIntersect(const std::set<int32_t> &left,
                   const std::set<int32_t> &right) {
  auto leftIt = left.begin();
  auto rightIt = right.begin();
  while (leftIt != left.end() && rightIt != right.end()) {
    if (*leftIt == *rightIt)
      return true;
    if (*leftIt < *rightIt)
      ++leftIt;
    else
      ++rightIt;
  }
  return false;
}

struct PlannedValue {
  Value value;
  int32_t valueId;
  std::set<int32_t> resumePCs;
  int32_t slot = -1;
};

struct ObjectStorage {
  std::string state;
  std::string waiterFrame;
  std::string waiterPC;
  std::string payload;
};

int32_t colorValues(SmallVectorImpl<PlannedValue> &values) {
  struct Slot {
    Type type;
    SmallVector<unsigned> values;
  };
  SmallVector<Slot> slots;
  for (auto indexed : llvm::enumerate(values)) {
    PlannedValue &value = indexed.value();
    int32_t selected = -1;
    for (auto slot : llvm::enumerate(slots)) {
      if (slot.value().type != value.value.getType())
        continue;
      bool interferes = llvm::any_of(slot.value().values, [&](unsigned other) {
        return setsIntersect(value.resumePCs, values[other].resumePCs);
      });
      if (!interferes) {
        selected = static_cast<int32_t>(slot.index());
        slot.value().values.push_back(indexed.index());
        break;
      }
    }
    if (selected < 0) {
      selected = static_cast<int32_t>(slots.size());
      slots.push_back(
          {value.value.getType(), {static_cast<unsigned>(indexed.index())}});
    }
    value.slot = selected;
  }
  return static_cast<int32_t>(slots.size());
}

SmallVector<Type> asyncResults(FunctionType type) {
  SmallVector<Type> results;
  if (type.getNumResults() == 0) {
    results.push_back(async::TokenType::get(type.getContext()));
    return results;
  }
  for (Type result : type.getResults())
    results.push_back(async::ValueType::get(result));
  return results;
}

struct LowerHostToAsyncPass
    : impl::LowerHostToAsyncPassBase<LowerHostToAsyncPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool hasExecute = false;
    module.walk([&](async::ExecuteOp operation) {
      operation.emitError(
          "async.execute is not supported by the structured Udon Host model");
      hasExecute = true;
    });
    if (hasExecute) {
      signalPassFailure();
      return;
    }
    std::set<std::string> maySuspend;
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      bool hasAwait = false;
      function.walk([&](runtime::AwaitOp) { hasAwait = true; });
      auto effect = function->getAttrOfType<StringAttr>("vrc.async_effect");
      if ((effect && effect.getValue() == "may_suspend") || hasAwait)
        maySuspend.insert(function.getSymName().str());
    }

    bool changed = true;
    while (changed) {
      changed = false;
      for (func::FuncOp function : module.getOps<func::FuncOp>()) {
        if (maySuspend.count(function.getSymName().str()))
          continue;
        WalkResult result = function.walk([&](func::CallOp call) {
          if (!maySuspend.count(call.getCallee().str()))
            return WalkResult::advance();
          maySuspend.insert(function.getSymName().str());
          changed = true;
          return WalkResult::interrupt();
        });
        (void)result;
      }
    }
    if (maySuspend.empty())
      return;

    bool hasIndirectCall = false;
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      if (!maySuspend.count(function.getSymName().str()))
        continue;
      function.walk([&](func::CallIndirectOp call) {
        call.emitError("indirect calls from a may-suspend Host function are "
                       "not supported");
        hasIndirectCall = true;
      });
    }
    if (hasIndirectCall) {
      signalPassFailure();
      return;
    }

    std::map<std::string, unsigned> visitState;
    std::function<bool(StringRef)> visit = [&](StringRef name) {
      unsigned &state = visitState[name.str()];
      if (state == 2)
        return true;
      if (state == 1) {
        module.emitError("recursive may-suspend Host call graph is not "
                         "supported: ")
            << name;
        return false;
      }
      state = 1;
      auto function = module.lookupSymbol<func::FuncOp>(name);
      if (!function) {
        module.emitError("may-suspend Host function is missing: ") << name;
        return false;
      }
      bool ok = true;
      function.walk([&](func::CallOp call) {
        if (ok && maySuspend.count(call.getCallee().str()))
          ok = visit(call.getCallee());
      });
      state = 2;
      return ok;
    };
    for (const std::string &name : maySuspend)
      if (!visit(name)) {
        signalPassFailure();
        return;
      }

    std::map<std::string, SmallVector<Type>> convertedResults;
    SmallVector<func::FuncOp> functions;
    for (func::FuncOp function : module.getOps<func::FuncOp>())
      if (maySuspend.count(function.getSymName().str()))
        functions.push_back(function);

    for (func::FuncOp function : functions) {
      std::string name = function.getSymName().str();
      FunctionType originalType = function.getFunctionType();
      SmallVector<Type> results = asyncResults(originalType);
      convertedResults.emplace(name, results);

      function.setSymName(name + ".udoncl_sync");
      OpBuilder builder(function);
      auto asyncType = FunctionType::get(module.getContext(),
                                         originalType.getInputs(), results);
      auto asyncFunction =
          async::FuncOp::create(builder, function.getLoc(), name, asyncType);
      for (NamedAttribute attribute : function->getAttrs()) {
        StringRef attributeName = attribute.getName().strref();
        if (attributeName == SymbolTable::getSymbolAttrName() ||
            attributeName == "function_type" ||
            attributeName == SymbolTable::getVisibilityAttrName())
          continue;
        asyncFunction->setAttr(attribute.getName(), attribute.getValue());
      }
      if (auto visibility =
              function->getAttr(SymbolTable::getVisibilityAttrName()))
        asyncFunction->setAttr(SymbolTable::getVisibilityAttrName(),
                               visibility);
      asyncFunction.getBody().takeBody(function.getBody());

      SmallVector<func::ReturnOp> returns;
      asyncFunction.walk(
          [&](func::ReturnOp operation) { returns.push_back(operation); });
      for (func::ReturnOp operation : returns) {
        OpBuilder returnBuilder(operation);
        async::ReturnOp::create(returnBuilder, operation.getLoc(),
                                operation.getOperands());
        operation.erase();
      }
      function.erase();
    }

    SmallVector<func::CallOp> calls;
    module.walk([&](func::CallOp call) {
      if (maySuspend.count(call.getCallee().str()))
        calls.push_back(call);
    });
    for (func::CallOp call : calls) {
      auto parent = call->getParentOfType<async::FuncOp>();
      if (!parent) {
        call.emitError(
            "may-suspend call must belong to a converted async.func");
        signalPassFailure();
        return;
      }
      OpBuilder builder(call);
      auto resultTypes = convertedResults.find(call.getCallee().str());
      if (resultTypes == convertedResults.end()) {
        call.emitError("converted async callee result types are missing");
        signalPassFailure();
        return;
      }
      auto asyncCall =
          async::CallOp::create(builder, call.getLoc(), call.getCallee(),
                                resultTypes->second, call.getOperands());
      if (call.getNumResults() == 0) {
        async::AwaitOp::create(builder, call.getLoc(), asyncCall.getResult(0));
      } else {
        for (auto [result, asyncResult] :
             llvm::zip(call.getResults(), asyncCall.getResults())) {
          auto awaited =
              async::AwaitOp::create(builder, call.getLoc(), asyncResult);
          result.replaceAllUsesWith(awaited.getResult());
        }
      }
      call.erase();
    }

    SmallVector<runtime::AwaitOp> awaits;
    module.walk(
        [&](runtime::AwaitOp operation) { awaits.push_back(operation); });
    for (runtime::AwaitOp operation : awaits) {
      if (!operation->getParentOfType<async::FuncOp>()) {
        operation.emitError("vrc.await must belong to a converted async.func");
        signalPassFailure();
        return;
      }
      OpBuilder builder(operation);
      Type valueType = async::ValueType::get(builder.getI32Type());
      auto value = runtime::EventAsAsyncOp::create(
          builder, operation.getLoc(), valueType, operation.getEvent());
      if (operation->hasAttr("vrc.cooperative_checkpoint"))
        value->setAttr("vrc.cooperative_checkpoint", builder.getUnitAttr());
      auto awaited =
          async::AwaitOp::create(builder, operation.getLoc(), value.getValue());
      operation.getStatus().replaceAllUsesWith(awaited.getResult());
      operation.erase();
    }
  }
};

struct LowerHostAsyncToNativePass
    : impl::LowerHostAsyncToNativePassBase<LowerHostAsyncToNativePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = module.getContext();
    Type nativeEvent = async::ValueType::get(IntegerType::get(context, 32));
    auto convertType = [&](Type type) -> Type {
      return isa<runtime::EventType>(type) ? nativeEvent : type;
    };

    module.walk([&](Operation *operation) {
      for (Region &region : operation->getRegions())
        for (Block &block : region)
          for (BlockArgument argument : block.getArguments())
            argument.setType(convertType(argument.getType()));
    });
    for (async::FuncOp function : module.getOps<async::FuncOp>()) {
      FunctionType type = function.getFunctionType();
      SmallVector<Type> inputs, results;
      llvm::transform(type.getInputs(), std::back_inserter(inputs),
                      convertType);
      llvm::transform(type.getResults(), std::back_inserter(results),
                      convertType);
      function->setAttr("function_type", TypeAttr::get(FunctionType::get(
                                             context, inputs, results)));
    }
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      FunctionType type = function.getFunctionType();
      SmallVector<Type> inputs, results;
      llvm::transform(type.getInputs(), std::back_inserter(inputs),
                      convertType);
      llvm::transform(type.getResults(), std::back_inserter(results),
                      convertType);
      function.setFunctionType(FunctionType::get(context, inputs, results));
    }

    SmallVector<runtime::EventAsAsyncOp> conversions;
    module.walk([&](runtime::EventAsAsyncOp operation) {
      conversions.push_back(operation);
    });
    bool invalid = false;
    for (runtime::EventAsAsyncOp operation : conversions) {
      Value event = operation->getOperand(0);
      if (event.getType() != nativeEvent || operation->getNumResults() != 1 ||
          operation->getResult(0).getType() != nativeEvent) {
        operation.emitError("Native Host event conversion requires an i32 "
                            "async value payload");
        invalid = true;
        continue;
      }
      operation->getResult(0).replaceAllUsesWith(event);
      operation.erase();
    }
    module.walk([&](Operation *operation) {
      for (Type type : operation->getOperandTypes())
        if (isa<runtime::EventType>(type)) {
          operation->emitError("VRC event operand remains after Native Host "
                               "Async conversion");
          invalid = true;
        }
      for (Type type : operation->getResultTypes())
        if (isa<runtime::EventType>(type)) {
          operation->emitError("VRC event result remains after Native Host "
                               "Async conversion");
          invalid = true;
        }
    });
    if (invalid)
      signalPassFailure();
  }
};

struct PrepareUdonAsyncRuntimePass
    : impl::PrepareUdonAsyncRuntimePassBase<PrepareUdonAsyncRuntimePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    const std::set<std::string> supportedOperations = {
        "async.coro.begin",
        "async.coro.end",
        "async.coro.free",
        "async.coro.id",
        "async.coro.save",
        "async.coro.suspend",
        "async.runtime.await_and_resume",
        "async.runtime.create",
        "async.runtime.is_error",
        "async.runtime.load",
        "async.runtime.set_available",
        "async.runtime.set_error",
        "async.runtime.store",
    };

    bool invalid = false;
    module.walk([&](Operation *operation) {
      StringRef name = operation->getName().getStringRef();
      if (operation->getName().getDialectNamespace() == "async" &&
          !supportedOperations.count(name.str())) {
        operation->emitError("operation is outside the supported structured "
                             "Udon Async Runtime subset");
        invalid = true;
      }
    });

    SmallVector<func::FuncOp> coroutines;
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      unsigned ids = 0, begins = 0, ends = 0, creates = 0;
      function.walk([&](Operation *operation) {
        StringRef name = operation->getName().getStringRef();
        ids += name == "async.coro.id";
        begins += name == "async.coro.begin";
        ends += name == "async.coro.end";
        creates += name == "async.runtime.create";
        if (isa<func::CallIndirectOp>(operation)) {
          operation->emitError("indirect calls are not supported in a Udon "
                               "coroutine");
          invalid = true;
        }
      });
      if (ids == 0)
        continue;
      if (ids != 1 || begins != 1 || ends != 1 || creates != 1) {
        function.emitError("Udon coroutine requires exactly one coro.id, "
                           "coro.begin, coro.end, and runtime.create");
        invalid = true;
      }
      coroutines.push_back(function);
    }

    llvm::sort(coroutines, [](func::FuncOp left, func::FuncOp right) {
      return left.getSymName() < right.getSymName();
    });
    std::set<std::string> coroutineNames;
    for (func::FuncOp function : coroutines)
      coroutineNames.insert(function.getSymName().str());

    for (func::FuncOp function : coroutines) {
      function.walk([&](func::CallOp call) {
        if (!coroutineNames.count(call.getCallee().str()))
          return;
        for (Value result : call.getResults()) {
          SmallVector<Operation *> awaits;
          for (OpOperand &use : result.getUses()) {
            Operation *owner = use.getOwner();
            StringRef name = owner->getName().getStringRef();
            if (name == "async.runtime.await_and_resume")
              awaits.push_back(owner);
            else if (name != "async.runtime.is_error" &&
                     name != "async.runtime.load") {
              owner->emitError("coroutine result escapes the supported "
                               "structured Udon call/await sequence");
              invalid = true;
            }
          }
          if (awaits.size() != 1) {
            call.emitError("coroutine result requires exactly one structured "
                           "await before another coroutine call");
            invalid = true;
            continue;
          }
          Operation *await = awaits.front();
          if (await->getBlock() != call->getBlock() ||
              !call->isBeforeInBlock(await)) {
            await->emitError("coroutine await must follow its call in the "
                             "same structured Host block");
            invalid = true;
            continue;
          }
          for (Operation *operation = call->getNextNode();
               operation && operation != await;
               operation = operation->getNextNode()) {
            auto nestedCall = dyn_cast<func::CallOp>(operation);
            if (nestedCall &&
                coroutineNames.count(nestedCall.getCallee().str())) {
              nestedCall.emitError("parallel coroutine calls are not "
                                   "supported by the Udon Host model");
              invalid = true;
              break;
            }
          }
        }
      });
    }
    if (invalid) {
      signalPassFailure();
      return;
    }

    std::map<std::string, unsigned> visitState;
    std::function<bool(func::FuncOp)> visit = [&](func::FuncOp function) {
      unsigned &state = visitState[function.getSymName().str()];
      if (state == 2)
        return true;
      if (state == 1) {
        function.emitError("recursive Udon coroutine call graph is not "
                           "supported");
        return false;
      }
      state = 1;
      bool ok = true;
      function.walk([&](func::CallOp call) {
        if (!ok || !coroutineNames.count(call.getCallee().str()))
          return;
        auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee());
        ok = callee && visit(callee);
      });
      state = 2;
      return ok;
    };
    for (func::FuncOp function : coroutines)
      if (!visit(function))
        invalid = true;

    if (invalid) {
      signalPassFailure();
      return;
    }

    DenseMap<Value, Value> objectParents;
    auto addObject = [&](Value value) {
      if (isAsyncObjectType(value.getType()))
        objectParents.try_emplace(value, value);
    };
    std::function<Value(Value)> findObject = [&](Value value) -> Value {
      auto found = objectParents.find(value);
      if (found == objectParents.end())
        return {};
      if (found->second == value)
        return value;
      found->second = findObject(found->second);
      return found->second;
    };
    auto unionObjects = [&](Value left, Value right) {
      addObject(left);
      addObject(right);
      Value leftRoot = findObject(left);
      Value rightRoot = findObject(right);
      if (leftRoot && rightRoot && leftRoot != rightRoot)
        objectParents[rightRoot] = leftRoot;
    };

    for (func::FuncOp function : coroutines) {
      for (BlockArgument argument : function.getArguments())
        addObject(argument);
      function.walk([&](Operation *operation) {
        for (Value result : operation->getResults())
          addObject(result);
      });
    }
    for (func::FuncOp function : coroutines) {
      function.walk([&](func::CallOp call) {
        if (!coroutineNames.count(call.getCallee().str()))
          return;
        auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee());
        if (!callee) {
          call.emitError("coroutine callee is missing during object identity "
                         "analysis");
          invalid = true;
          return;
        }
        for (auto [operand, argument] :
             llvm::zip(call.getOperands(), callee.getArguments()))
          if (isAsyncObjectType(operand.getType()) &&
              isAsyncObjectType(argument.getType()))
            unionObjects(operand, argument);

        SmallVector<func::ReturnOp> returns;
        callee.walk(
            [&](func::ReturnOp operation) { returns.push_back(operation); });
        for (func::ReturnOp operation : returns) {
          if (operation.getNumOperands() != call.getNumResults()) {
            call.emitError("coroutine return/result arity mismatch during "
                           "object identity analysis");
            invalid = true;
            continue;
          }
          for (auto [result, returned] :
               llvm::zip(call.getResults(), operation.getOperands()))
            if (isAsyncObjectType(result.getType()) &&
                isAsyncObjectType(returned.getType()))
              unionObjects(result, returned);
        }
      });
    }
    if (invalid) {
      signalPassFailure();
      return;
    }

    DenseMap<Value, int32_t> objectIds;
    DenseMap<Value, int32_t> rootIds;
    std::map<int32_t, Type> objectTypes;
    int32_t nextObjectId = 1;
    auto assignObjectId = [&](Value value) {
      if (!isAsyncObjectType(value.getType()))
        return;
      Value root = findObject(value);
      auto found = rootIds.find(root);
      int32_t id = 0;
      if (found == rootIds.end()) {
        id = nextObjectId++;
        rootIds.try_emplace(root, id);
        objectTypes.try_emplace(id, value.getType());
      } else {
        id = found->second;
        if (objectTypes.at(id) != value.getType()) {
          module.emitError("aliased async object types do not match");
          invalid = true;
        }
      }
      objectIds[value] = id;
    };
    for (func::FuncOp function : coroutines) {
      for (Block &block : function.getBody()) {
        for (BlockArgument argument : block.getArguments())
          assignObjectId(argument);
        for (Operation &operation : block)
          for (Value result : operation.getResults())
            assignObjectId(result);
      }
    }
    if (invalid) {
      signalPassFailure();
      return;
    }

    std::set<int32_t> checkpointObjectIds;
    for (func::FuncOp function : coroutines)
      function.walk([&](runtime::EventAsAsyncOp operation) {
        if (!operation->hasAttr("vrc.cooperative_checkpoint"))
          return;
        auto found = objectIds.find(operation.getResult());
        if (found == objectIds.end()) {
          operation.emitError("cooperative checkpoint has no object id");
          invalid = true;
          return;
        }
        checkpointObjectIds.insert(found->second);
      });
    if (invalid) {
      signalPassFailure();
      return;
    }

    Builder builder(module.getContext());
    module->setAttr(
        "vrc.async_object_state_encoding",
        builder.getDictionaryAttr(
            {builder.getNamedAttr("pending", builder.getI32IntegerAttr(0)),
             builder.getNamedAttr("available", builder.getI32IntegerAttr(1)),
             builder.getNamedAttr("error", builder.getI32IntegerAttr(-1))}));
    std::map<int32_t, ObjectStorage> objectStorage;
    SmallVector<Attribute> resumeRoutes;
    SmallVector<Attribute> checkpointRoutes;
    SmallVector<Attribute> objectResumeRoutes;
    for (auto indexed : llvm::enumerate(coroutines)) {
      func::FuncOp function = indexed.value();
      int32_t frameId = static_cast<int32_t>(indexed.index() + 1);
      function->setAttr("vrc.async_frame_id",
                        builder.getI32IntegerAttr(frameId));
      OpBuilder globalBuilder(function);
      auto createGlobal = [&](StringRef symbol, Type storageType) -> bool {
        if (module.lookupSymbol(symbol)) {
          function.emitError("generated async frame symbol collides: ")
              << symbol;
          invalid = true;
          return false;
        }
        Attribute initial = zeroFrameValue(storageType, builder);
        if (!initial) {
          function.emitError("cannot materialize async frame storage type: ")
              << storageType;
          invalid = true;
          return false;
        }
        ::vrc::udonhl::GlobalOp::create(
            globalBuilder, function.getLoc(), symbol, storageType, initial,
            "internal", 0, frameAlignment(storageType), false);
        return true;
      };
      std::string pcSymbol =
          "__udoncl_async_frame_" + std::to_string(frameId) + "_pc";
      createGlobal(pcSymbol, builder.getI32Type());
      function->setAttr("vrc.async_pc_symbol",
                        FlatSymbolRefAttr::get(module.getContext(), pcSymbol));
      int32_t resumePC = 1;
      function.walk([&](Operation *operation) {
        if (operation->getName().getStringRef() != "async.coro.suspend")
          return;
        int32_t pc = resumePC++;
        operation->setAttr("vrc.async_resume_pc",
                           builder.getI32IntegerAttr(pc));
        OpBuilder pcBuilder(operation);
        auto value = arith::ConstantIntOp::create(pcBuilder,
                                                  operation->getLoc(), pc, 32);
        auto address = ::vrc::udonhl::AddressOfOp::create(
            pcBuilder, operation->getLoc(),
            ::vrc::udonhl::PtrType::get(module.getContext(), 0), pcSymbol);
        auto store = ::vrc::udonhl::StoreOp::create(
            pcBuilder, operation->getLoc(), value, address.getResult(), 4);
        store->setAttr("vrc.async_pc_store", builder.getI32IntegerAttr(pc));
      });
      function->setAttr("vrc.async_resume_count",
                        builder.getI32IntegerAttr(resumePC - 1));

      DenseMap<Value, int32_t> valueIds;
      int32_t nextValueId = 0;
      for (Block &block : function.getBody()) {
        for (BlockArgument argument : block.getArguments())
          valueIds.try_emplace(argument, nextValueId++);
        for (Operation &operation : block)
          for (Value result : operation.getResults())
            valueIds.try_emplace(result, nextValueId++);
      }

      Liveness liveness(function);
      DenseMap<Value, std::set<int32_t>> spillResumePCs;
      DenseMap<Value, std::set<int32_t>> objectResumePCs;
      SmallVector<Operation *> suspends;
      function.walk([&](Operation *operation) {
        if (operation->getName().getStringRef() != "async.coro.suspend")
          return;
        suspends.push_back(operation);
        int32_t pc = static_cast<int32_t>(
            operation->getAttrOfType<IntegerAttr>("vrc.async_resume_pc")
                .getInt());
        const LivenessBlockInfo *blockInfo =
            liveness.getLiveness(operation->getBlock());
        if (!blockInfo) {
          operation->emitError("missing liveness information for Udon "
                               "suspend planning");
          invalid = true;
          return;
        }
        for (Value value : blockInfo->currentlyLiveValues(operation)) {
          if (isRecomputableAcrossSuspend(value) ||
              isCoroutineBookkeeping(value))
            continue;
          if (isAsyncObjectType(value.getType())) {
            objectResumePCs[value].insert(pc);
            continue;
          }
          if (!isSupportedSpillType(value.getType())) {
            operation->emitError("unsupported live-across-suspend value type: ")
                << value.getType();
            invalid = true;
            continue;
          }
          spillResumePCs[value].insert(pc);
        }
      });
      if (invalid)
        continue;

      auto makePlan = [&](const DenseMap<Value, std::set<int32_t>> &liveSets,
                          StringRef planAttr, StringRef countAttr,
                          StringRef liveAttr, bool annotateResultIds) {
        SmallVector<PlannedValue> values;
        values.reserve(liveSets.size());
        for (const auto &entry : liveSets)
          values.push_back(
              {entry.first, valueIds.lookup(entry.first), entry.second, -1});
        llvm::sort(values,
                   [](const PlannedValue &left, const PlannedValue &right) {
                     return left.valueId < right.valueId;
                   });
        int32_t slotCount = 0;
        SmallVector<Type> slotTypes;
        SmallVector<std::string> primarySymbols;
        if (!annotateResultIds) {
          slotCount = colorValues(values);
          slotTypes.resize(slotCount);
          primarySymbols.resize(slotCount);
          for (const PlannedValue &value : values)
            slotTypes[value.slot] = value.value.getType();
          for (int32_t slot = 0; slot < slotCount; ++slot) {
            std::string base = "__udoncl_async_frame_" +
                               std::to_string(frameId) + "_spill_" +
                               std::to_string(slot);
            primarySymbols[slot] = base;
            createGlobal(
                base, physicalFrameType(slotTypes[slot], module.getContext()));
          }
        } else {
          std::set<int32_t> functionObjectIds;
          for (PlannedValue &value : values) {
            auto found = objectIds.find(value.value);
            if (found == objectIds.end()) {
              function.emitError("live async object has no canonical identity");
              invalid = true;
              continue;
            }
            value.slot = found->second;
            functionObjectIds.insert(value.slot);
          }
          slotCount = static_cast<int32_t>(functionObjectIds.size());
          for (int32_t objectId : functionObjectIds) {
            if (checkpointObjectIds.count(objectId) ||
                objectStorage.count(objectId))
              continue;
            Type objectType = objectTypes.at(objectId);
            std::string base =
                "__udoncl_async_object_" + std::to_string(objectId);
            ObjectStorage storage{base + "_state",
                                  base + "_waiter_frame",
                                  base + "_waiter_pc",
                                  {}};
            createGlobal(storage.state, builder.getI32Type());
            createGlobal(storage.waiterFrame, builder.getI32Type());
            createGlobal(storage.waiterPC, builder.getI32Type());
            if (auto valueType = dyn_cast<async::ValueType>(objectType)) {
              Type payloadType = physicalFrameType(valueType.getValueType(),
                                                   module.getContext());
              if (!isSupportedSpillType(payloadType)) {
                function.emitError("unsupported async object payload type: ")
                    << payloadType;
                invalid = true;
                continue;
              }
              storage.payload = base + "_payload";
              createGlobal(storage.payload, payloadType);
            }
            objectStorage.try_emplace(objectId, std::move(storage));
          }
        }
        if (invalid)
          return;
        SmallVector<Attribute> plan;
        DenseMap<Value, int32_t> slotsByValue;
        for (const PlannedValue &value : values) {
          slotsByValue[value.value] = value.slot;
          SmallVector<int32_t> pcs(value.resumePCs.begin(),
                                   value.resumePCs.end());
          NamedAttrList attributes;
          attributes.append("value_id",
                            builder.getI32IntegerAttr(value.valueId));
          attributes.append(annotateResultIds ? "object_id" : "slot",
                            builder.getI32IntegerAttr(value.slot));
          attributes.append(
              "type", builder.getStringAttr(typeName(value.value.getType())));
          attributes.append("resume_pcs", builder.getDenseI32ArrayAttr(pcs));
          if (!annotateResultIds)
            attributes.append(
                "symbol", FlatSymbolRefAttr::get(module.getContext(),
                                                 primarySymbols[value.slot]));
          else {
            if (checkpointObjectIds.count(value.slot)) {
              attributes.append("cooperative_checkpoint",
                                builder.getUnitAttr());
              plan.push_back(builder.getDictionaryAttr(attributes));
              continue;
            }
            const ObjectStorage &storage = objectStorage.at(value.slot);
            attributes.append(
                "state_symbol",
                FlatSymbolRefAttr::get(module.getContext(), storage.state));
            attributes.append("waiter_frame_symbol",
                              FlatSymbolRefAttr::get(module.getContext(),
                                                     storage.waiterFrame));
            attributes.append(
                "waiter_pc_symbol",
                FlatSymbolRefAttr::get(module.getContext(), storage.waiterPC));
            if (!storage.payload.empty())
              attributes.append(
                  "payload_symbol",
                  FlatSymbolRefAttr::get(module.getContext(), storage.payload));
          }
          plan.push_back(builder.getDictionaryAttr(attributes));
        }
        function->setAttr(planAttr, builder.getArrayAttr(plan));
        function->setAttr(countAttr, builder.getI32IntegerAttr(slotCount));

        for (Operation *suspend : suspends) {
          int32_t pc = static_cast<int32_t>(
              suspend->getAttrOfType<IntegerAttr>("vrc.async_resume_pc")
                  .getInt());
          SmallVector<int32_t> liveSlots;
          for (const PlannedValue &value : values)
            if (value.resumePCs.count(pc))
              liveSlots.push_back(value.slot);
          llvm::sort(liveSlots);
          liveSlots.erase(std::unique(liveSlots.begin(), liveSlots.end()),
                          liveSlots.end());
          suspend->setAttr(liveAttr, builder.getDenseI32ArrayAttr(liveSlots));
        }

        if (!annotateResultIds) {
          DenseMap<int32_t, Operation *> suspendByPC;
          for (Operation *suspend : suspends) {
            int32_t pc = static_cast<int32_t>(
                suspend->getAttrOfType<IntegerAttr>("vrc.async_resume_pc")
                    .getInt());
            suspendByPC.try_emplace(pc, suspend);
          }
          Type pointer = ::vrc::udonhl::PtrType::get(module.getContext(), 0);
          for (const PlannedValue &value : values) {
            for (int32_t pc : value.resumePCs) {
              Operation *suspend = suspendByPC.lookup(pc);
              if (!suspend || suspend->getNumSuccessors() < 2) {
                function.emitError(
                    "invalid coroutine suspend successor layout");
                invalid = true;
                break;
              }
              OpBuilder spillBuilder(suspend);
              auto address = ::vrc::udonhl::AddressOfOp::create(
                  spillBuilder, suspend->getLoc(), pointer,
                  primarySymbols[value.slot]);
              auto store = ::vrc::udonhl::StoreOp::create(
                  spillBuilder, suspend->getLoc(), value.value,
                  address.getResult(),
                  frameAlignment(physicalFrameType(value.value.getType(),
                                                   module.getContext())));
              store->setAttr("vrc.async_spill_pc",
                             builder.getI32IntegerAttr(pc));
            }
            if (invalid)
              break;

            SmallVector<::vrc::udonhl::LoadOp> reloads;
            for (int32_t pc : value.resumePCs) {
              Operation *suspend = suspendByPC.lookup(pc);
              Block *resume = suspend->getSuccessor(1);
              OpBuilder reloadBuilder = OpBuilder::atBlockBegin(resume);
              auto address = ::vrc::udonhl::AddressOfOp::create(
                  reloadBuilder, suspend->getLoc(), pointer,
                  primarySymbols[value.slot]);
              auto load = ::vrc::udonhl::LoadOp::create(
                  reloadBuilder, suspend->getLoc(), value.value.getType(),
                  address.getResult(),
                  frameAlignment(physicalFrameType(value.value.getType(),
                                                   module.getContext())));
              load->setAttr("vrc.async_reload_pc",
                            builder.getI32IntegerAttr(pc));
              reloads.push_back(load);
            }
            DominanceInfo dominance(function);
            for (OpOperand &use :
                 llvm::make_early_inc_range(value.value.getUses())) {
              Operation *selected = nullptr;
              for (::vrc::udonhl::LoadOp reload : reloads) {
                Operation *candidate = reload.getOperation();
                if (!dominance.dominates(candidate, use.getOwner()))
                  continue;
                if (!selected || dominance.dominates(selected, candidate))
                  selected = candidate;
              }
              if (selected)
                use.set(cast<::vrc::udonhl::LoadOp>(selected).getResult());
            }
          }
          return;
        }
        function.walk([&](Operation *operation) {
          SmallVector<int32_t> resultIds(operation->getNumResults(), -1);
          bool hasObject = false;
          for (auto result : llvm::enumerate(operation->getResults())) {
            auto found = slotsByValue.find(result.value());
            if (found == slotsByValue.end())
              continue;
            resultIds[result.index()] = found->second;
            hasObject = true;
          }
          if (hasObject)
            operation->setAttr("vrc.async_result_object_ids",
                               builder.getDenseI32ArrayAttr(resultIds));
        });
      };

      makePlan(spillResumePCs, "vrc.async_spill_slots",
               "vrc.async_spill_slot_count", "vrc.async_live_spill_slots",
               false);
      makePlan(objectResumePCs, "vrc.async_objects", "vrc.async_object_count",
               "vrc.async_live_object_ids", true);
      if (invalid)
        continue;

      function.walk([&](func::ReturnOp operation) {
        SmallVector<int32_t> returnedObjectIds;
        returnedObjectIds.reserve(operation.getNumOperands());
        for (Value operand : operation.getOperands()) {
          auto found = objectIds.find(operand);
          returnedObjectIds.push_back(found == objectIds.end() ? -1
                                                               : found->second);
        }
        operation->setAttr("vrc.async_return_object_ids",
                           builder.getDenseI32ArrayAttr(returnedObjectIds));
      });

      auto storageFor = [&](Value object) -> ObjectStorage * {
        auto id = objectIds.find(object);
        if (id == objectIds.end()) {
          function.emitError("async runtime object has no canonical identity");
          invalid = true;
          return nullptr;
        }
        auto storage = objectStorage.find(id->second);
        if (storage == objectStorage.end()) {
          function.emitError(
              "async runtime object storage was not materialized");
          invalid = true;
          return nullptr;
        }
        return &storage->second;
      };
      auto checkpointIdFor = [&](Value object) -> std::optional<int32_t> {
        auto found = objectIds.find(object);
        if (found == objectIds.end() ||
            !checkpointObjectIds.count(found->second))
          return std::nullopt;
        return found->second;
      };
      auto emitStore = [&](OpBuilder &storeBuilder, Location location,
                           Value value, StringRef symbol, StringRef marker) {
        auto address = ::vrc::udonhl::AddressOfOp::create(
            storeBuilder, location,
            ::vrc::udonhl::PtrType::get(module.getContext(), 0), symbol);
        auto store = ::vrc::udonhl::StoreOp::create(
            storeBuilder, location, value, address.getResult(),
            frameAlignment(
                physicalFrameType(value.getType(), module.getContext())));
        store->setAttr(marker, builder.getUnitAttr());
      };
      auto emitI32Store = [&](OpBuilder &storeBuilder, Location location,
                              int32_t value, StringRef symbol,
                              StringRef marker) {
        auto constant =
            arith::ConstantIntOp::create(storeBuilder, location, value, 32);
        emitStore(storeBuilder, location, constant, symbol, marker);
      };

      SmallVector<Operation *> runtimeOperations;
      function.walk([&](Operation *operation) {
        StringRef name = operation->getName().getStringRef();
        if (name == "async.runtime.create" || name == "vrc.event_as_async" ||
            name == "async.runtime.store" || name == "async.runtime.load" ||
            name == "async.runtime.is_error" ||
            name == "async.runtime.set_available" ||
            name == "async.runtime.set_error" ||
            name == "async.runtime.await_and_resume")
          runtimeOperations.push_back(operation);
      });
      for (Operation *operation : runtimeOperations) {
        StringRef name = operation->getName().getStringRef();
        if (name == "async.runtime.create" || name == "vrc.event_as_async") {
          if (operation->getNumResults() != 1)
            continue;
          if (checkpointIdFor(operation->getResult(0)))
            continue;
          ObjectStorage *storage = storageFor(operation->getResult(0));
          if (!storage)
            continue;
          OpBuilder resetBuilder(operation);
          resetBuilder.setInsertionPointAfter(operation);
          emitI32Store(resetBuilder, operation->getLoc(), 0, storage->state,
                       "vrc.async_object_state_reset");
          emitI32Store(resetBuilder, operation->getLoc(), 0,
                       storage->waiterFrame, "vrc.async_object_waiter_reset");
          emitI32Store(resetBuilder, operation->getLoc(), 0, storage->waiterPC,
                       "vrc.async_object_waiter_reset");
          continue;
        }
        if (name == "async.runtime.store") {
          if (operation->getNumOperands() != 2)
            continue;
          ObjectStorage *storage = storageFor(operation->getOperand(1));
          if (!storage || storage->payload.empty()) {
            operation->emitError("async.value store has no payload storage");
            invalid = true;
            continue;
          }
          OpBuilder payloadBuilder(operation);
          emitStore(payloadBuilder, operation->getLoc(),
                    operation->getOperand(0), storage->payload,
                    "vrc.async_object_payload_store");
          operation->erase();
          continue;
        }
        if (name == "async.runtime.load") {
          if (operation->getNumOperands() != 1 ||
              operation->getNumResults() != 1)
            continue;
          if (checkpointIdFor(operation->getOperand(0))) {
            OpBuilder payloadBuilder(operation);
            auto zero = arith::ConstantIntOp::create(
                payloadBuilder, operation->getLoc(), 0, 32);
            operation->getResult(0).replaceAllUsesWith(zero.getResult());
            operation->erase();
            continue;
          }
          ObjectStorage *storage = storageFor(operation->getOperand(0));
          if (!storage || storage->payload.empty()) {
            operation->emitError("async.value load has no payload storage");
            invalid = true;
            continue;
          }
          OpBuilder payloadBuilder(operation);
          auto address = ::vrc::udonhl::AddressOfOp::create(
              payloadBuilder, operation->getLoc(),
              ::vrc::udonhl::PtrType::get(module.getContext(), 0),
              storage->payload);
          auto load = ::vrc::udonhl::LoadOp::create(
              payloadBuilder, operation->getLoc(),
              operation->getResult(0).getType(), address.getResult(),
              frameAlignment(physicalFrameType(
                  operation->getResult(0).getType(), module.getContext())));
          load->setAttr("vrc.async_object_payload_load", builder.getUnitAttr());
          operation->getResult(0).replaceAllUsesWith(load.getResult());
          operation->erase();
          continue;
        }
        if (name == "async.runtime.is_error") {
          if (operation->getNumOperands() != 1 ||
              operation->getNumResults() != 1)
            continue;
          if (checkpointIdFor(operation->getOperand(0))) {
            OpBuilder stateBuilder(operation);
            auto notError = arith::ConstantIntOp::create(
                stateBuilder, operation->getLoc(), 0, 1);
            operation->getResult(0).replaceAllUsesWith(notError.getResult());
            operation->erase();
            continue;
          }
          ObjectStorage *storage = storageFor(operation->getOperand(0));
          if (!storage)
            continue;
          OpBuilder stateBuilder(operation);
          auto address = ::vrc::udonhl::AddressOfOp::create(
              stateBuilder, operation->getLoc(),
              ::vrc::udonhl::PtrType::get(module.getContext(), 0),
              storage->state);
          auto state = ::vrc::udonhl::LoadOp::create(
              stateBuilder, operation->getLoc(), builder.getI32Type(),
              address.getResult(), 4);
          state->setAttr("vrc.async_object_state_load", builder.getUnitAttr());
          auto error = arith::ConstantIntOp::create(
              stateBuilder, operation->getLoc(), -1, 32);
          auto isError = arith::CmpIOp::create(
              stateBuilder, operation->getLoc(), arith::CmpIPredicate::eq,
              state.getResult(), error);
          operation->getResult(0).replaceAllUsesWith(isError.getResult());
          operation->erase();
          continue;
        }
        if (name == "async.runtime.set_available" ||
            name == "async.runtime.set_error") {
          if (operation->getNumOperands() != 1)
            continue;
          ObjectStorage *storage = storageFor(operation->getOperand(0));
          if (!storage)
            continue;
          OpBuilder stateBuilder(operation);
          emitI32Store(stateBuilder, operation->getLoc(),
                       name == "async.runtime.set_available" ? 1 : -1,
                       storage->state,
                       name == "async.runtime.set_available"
                           ? "vrc.async_object_set_available"
                           : "vrc.async_object_set_error");
          auto waiterFrameAddress = ::vrc::udonhl::AddressOfOp::create(
              stateBuilder, operation->getLoc(),
              ::vrc::udonhl::PtrType::get(module.getContext(), 0),
              storage->waiterFrame);
          auto waiterFrame = ::vrc::udonhl::LoadOp::create(
              stateBuilder, operation->getLoc(), builder.getI32Type(),
              waiterFrameAddress.getResult(), 4);
          waiterFrame->setAttr("vrc.async_object_waiter_frame_load",
                               builder.getUnitAttr());
          auto waiterPCAddress = ::vrc::udonhl::AddressOfOp::create(
              stateBuilder, operation->getLoc(),
              ::vrc::udonhl::PtrType::get(module.getContext(), 0),
              storage->waiterPC);
          auto waiterPC = ::vrc::udonhl::LoadOp::create(
              stateBuilder, operation->getLoc(), builder.getI32Type(),
              waiterPCAddress.getResult(), 4);
          waiterPC->setAttr("vrc.async_object_waiter_pc_load",
                            builder.getUnitAttr());
          emitI32Store(stateBuilder, operation->getLoc(), 0,
                       storage->waiterFrame, "vrc.async_object_waiter_clear");
          emitI32Store(stateBuilder, operation->getLoc(), 0, storage->waiterPC,
                       "vrc.async_object_waiter_clear");
          OperationState resumeState(
              operation->getLoc(),
              ::vrc::udonhl::AsyncResumeOp::getOperationName());
          resumeState.addOperands(
              {waiterFrame.getResult(), waiterPC.getResult()});
          auto objectId = objectIds.find(operation->getOperand(0));
          if (objectId != objectIds.end())
            resumeState.addAttribute(
                "vrc.async_object_id",
                stateBuilder.getI32IntegerAttr(objectId->second));
          stateBuilder.create(resumeState);
          operation->erase();
          continue;
        }
        if (name == "async.runtime.await_and_resume") {
          if (operation->getNumOperands() < 1)
            continue;
          std::optional<int32_t> checkpointId =
              checkpointIdFor(operation->getOperand(0));
          ObjectStorage *storage =
              checkpointId ? nullptr : storageFor(operation->getOperand(0));
          if (!checkpointId && !storage)
            continue;
          Operation *suspend = operation->getNextNode();
          while (suspend &&
                 suspend->getName().getStringRef() != "async.coro.suspend")
            suspend = suspend->getNextNode();
          auto pc =
              suspend
                  ? suspend->getAttrOfType<IntegerAttr>("vrc.async_resume_pc")
                  : IntegerAttr();
          if (!pc) {
            operation->emitError("await_and_resume has no following suspend");
            invalid = true;
            continue;
          }
          if (checkpointId) {
            suspend->setAttr("vrc.cooperative_checkpoint_object_id",
                             builder.getI32IntegerAttr(*checkpointId));
            Block *current = operation->getBlock();
            Region *region = current->getParent();
            Block *pending = new Block();
            region->getBlocks().insert(std::next(current->getIterator()),
                                       pending);
            suspend->moveBefore(pending, pending->end());
            cf::BranchOp::create(OpBuilder::atBlockEnd(current),
                                 operation->getLoc(), pending);
            operation->erase();
            continue;
          }
          auto objectId = objectIds.find(operation->getOperand(0));
          if (objectId == objectIds.end()) {
            operation->emitError("awaited async object has no identity");
            invalid = true;
            continue;
          }
          NamedAttrList objectRoute;
          objectRoute.append("object_id",
                             builder.getI32IntegerAttr(objectId->second));
          objectRoute.append("frame_id", builder.getI32IntegerAttr(frameId));
          objectRoute.append("resume_pc", pc);
          objectResumeRoutes.push_back(builder.getDictionaryAttr(objectRoute));
          OpBuilder waiterBuilder(operation);
          emitI32Store(waiterBuilder, operation->getLoc(), frameId,
                       storage->waiterFrame,
                       "vrc.async_object_waiter_frame_store");
          emitI32Store(waiterBuilder, operation->getLoc(),
                       static_cast<int32_t>(pc.getInt()), storage->waiterPC,
                       "vrc.async_object_waiter_pc_store");
          Block *current = operation->getBlock();
          Block *resume = suspend->getSuccessor(1);
          Region *region = current->getParent();
          Block *pending = new Block();
          region->getBlocks().insert(std::next(current->getIterator()),
                                     pending);
          suspend->moveBefore(pending, pending->end());

          OpBuilder branchBuilder = OpBuilder::atBlockEnd(current);
          auto stateAddress = ::vrc::udonhl::AddressOfOp::create(
              branchBuilder, operation->getLoc(),
              ::vrc::udonhl::PtrType::get(module.getContext(), 0),
              storage->state);
          auto state = ::vrc::udonhl::LoadOp::create(
              branchBuilder, operation->getLoc(), builder.getI32Type(),
              stateAddress.getResult(), 4);
          state->setAttr("vrc.async_object_ready_state_load",
                         builder.getUnitAttr());
          auto pendingState = arith::ConstantIntOp::create(
              branchBuilder, operation->getLoc(), 0, 32);
          auto ready = arith::CmpIOp::create(branchBuilder, operation->getLoc(),
                                             arith::CmpIPredicate::ne,
                                             state.getResult(), pendingState);
          cf::CondBranchOp::create(branchBuilder, operation->getLoc(), ready,
                                   resume, ValueRange{}, pending, ValueRange{});
          operation->erase();
        }
      }
      std::set<int32_t> routedPCs;
      for (Operation *suspend : suspends) {
        auto pc = suspend->getAttrOfType<IntegerAttr>("vrc.async_resume_pc");
        if (!pc || pc.getInt() <= 0 ||
            !routedPCs.insert(static_cast<int32_t>(pc.getInt())).second ||
            suspend->getNumSuccessors() < 2) {
          suspend->emitError("invalid or duplicate coroutine resume route");
          invalid = true;
          continue;
        }
        Block *resume = suspend->getSuccessor(1);
        OpBuilder markerBuilder = OpBuilder::atBlockBegin(resume);
        OperationState markerState(
            suspend->getLoc(),
            ::vrc::udonhl::AsyncResumePointOp::getOperationName());
        markerState.addAttribute("frame_id",
                                 builder.getI32IntegerAttr(frameId));
        markerState.addAttribute(
            "resume_pc",
            builder.getI32IntegerAttr(static_cast<int32_t>(pc.getInt())));
        markerBuilder.create(markerState);

        NamedAttrList route;
        route.append("frame_id", builder.getI32IntegerAttr(frameId));
        route.append("resume_pc", builder.getI32IntegerAttr(pc.getInt()));
        route.append("function", FlatSymbolRefAttr::get(module.getContext(),
                                                        function.getSymName()));
        route.append("pc_symbol",
                     FlatSymbolRefAttr::get(module.getContext(), pcSymbol));
        resumeRoutes.push_back(builder.getDictionaryAttr(route));
        if (auto objectId = suspend->getAttrOfType<IntegerAttr>(
                "vrc.cooperative_checkpoint_object_id")) {
          NamedAttrList checkpointRoute;
          checkpointRoute.append("object_id", objectId);
          checkpointRoute.append("frame_id",
                                 builder.getI32IntegerAttr(frameId));
          checkpointRoute.append("resume_pc", pc);
          checkpointRoutes.push_back(
              builder.getDictionaryAttr(checkpointRoute));
        }
      }
    }
    module->setAttr("vrc.async_resume_routes",
                    builder.getArrayAttr(resumeRoutes));
    module->setAttr("vrc.async_checkpoint_routes",
                    builder.getArrayAttr(checkpointRoutes));
    module->setAttr("vrc.async_object_resume_routes",
                    builder.getArrayAttr(objectResumeRoutes));
    if (invalid)
      signalPassFailure();
  }
};

struct LowerUdonAsyncRuntimePass
    : impl::LowerUdonAsyncRuntimePassBase<LowerUdonAsyncRuntimePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto routes = module->getAttrOfType<ArrayAttr>("vrc.async_resume_routes");
    if (!routes) {
      module.emitError("Udon Async lowering requires a prepared resume route "
                       "table");
      signalPassFailure();
      return;
    }
    auto checkpointRouteAttrs =
        module->getAttrOfType<ArrayAttr>("vrc.async_checkpoint_routes");
    DenseMap<int32_t, std::pair<int32_t, int32_t>> checkpointRoutes;
    if (checkpointRouteAttrs)
      for (Attribute attribute : checkpointRouteAttrs) {
        auto route = dyn_cast<DictionaryAttr>(attribute);
        auto objectId =
            route ? route.getAs<IntegerAttr>("object_id") : IntegerAttr();
        auto frameId =
            route ? route.getAs<IntegerAttr>("frame_id") : IntegerAttr();
        auto resumePC =
            route ? route.getAs<IntegerAttr>("resume_pc") : IntegerAttr();
        if (!objectId || !frameId || !resumePC || objectId.getInt() <= 0 ||
            frameId.getInt() <= 0 || resumePC.getInt() <= 0 ||
            !checkpointRoutes
                 .try_emplace(
                     static_cast<int32_t>(objectId.getInt()),
                     std::make_pair(static_cast<int32_t>(frameId.getInt()),
                                    static_cast<int32_t>(resumePC.getInt())))
                 .second) {
          module.emitError("invalid or duplicate cooperative checkpoint route");
          signalPassFailure();
          return;
        }
      }
    auto objectRouteAttrs =
        module->getAttrOfType<ArrayAttr>("vrc.async_object_resume_routes");
    std::map<int32_t, std::set<std::pair<int32_t, int32_t>>> objectResumeRoutes;
    if (objectRouteAttrs)
      for (Attribute attribute : objectRouteAttrs) {
        auto route = dyn_cast<DictionaryAttr>(attribute);
        auto objectId =
            route ? route.getAs<IntegerAttr>("object_id") : IntegerAttr();
        auto frameId =
            route ? route.getAs<IntegerAttr>("frame_id") : IntegerAttr();
        auto resumePC =
            route ? route.getAs<IntegerAttr>("resume_pc") : IntegerAttr();
        if (!objectId || !frameId || !resumePC || objectId.getInt() <= 0 ||
            frameId.getInt() <= 0 || resumePC.getInt() <= 0) {
          module.emitError("invalid async object resume route");
          signalPassFailure();
          return;
        }
        objectResumeRoutes[static_cast<int32_t>(objectId.getInt())].insert(
            {static_cast<int32_t>(frameId.getInt()),
             static_cast<int32_t>(resumePC.getInt())});
      }

    SmallVector<func::FuncOp> coroutines;
    std::set<std::string> coroutineNames;
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      if (!function->hasAttr("vrc.async_frame_id"))
        continue;
      coroutines.push_back(function);
      coroutineNames.insert(function.getSymName().str());
    }
    bool invalid = false;
    std::set<int32_t> boundObjectIds;
    std::set<int32_t> specializedObjectDispatchers;
    for (func::FuncOp function : coroutines) {
      SmallVector<Type> physicalResults;
      bool publicExport = function->hasAttr("vrc.export_id");
      SmallVector<func::ReturnOp> returns;
      function.walk(
          [&](func::ReturnOp operation) { returns.push_back(operation); });
      for (func::ReturnOp operation : returns) {
        OpBuilder returnBuilder(operation);
        SmallVector<Value> returnValues;
        if (publicExport && operation.getNumOperands() == 1) {
          auto valueType =
              dyn_cast<async::ValueType>(operation.getOperand(0).getType());
          auto objectIds = operation->getAttrOfType<DenseI32ArrayAttr>(
              "vrc.async_return_object_ids");
          if (!valueType || !objectIds || objectIds.size() != 1 ||
              objectIds[0] <= 0) {
            operation.emitError(
                "public async export result has no canonical object identity");
            invalid = true;
          } else {
            StringAttr payloadSymbol;
            auto objects =
                function->getAttrOfType<ArrayAttr>("vrc.async_objects");
            if (objects)
              for (Attribute attribute : objects) {
                auto object = dyn_cast<DictionaryAttr>(attribute);
                auto id = object ? object.getAs<IntegerAttr>("object_id")
                                 : IntegerAttr();
                if (!id || id.getInt() != objectIds[0])
                  continue;
                auto symbol = object.getAs<FlatSymbolRefAttr>("payload_symbol");
                if (symbol)
                  payloadSymbol =
                      returnBuilder.getStringAttr(symbol.getValue());
                break;
              }
            if (!payloadSymbol) {
              operation.emitError(
                  "public async export result has no payload storage");
              invalid = true;
            } else {
              Type resultType = valueType.getValueType();
              auto address = ::vrc::udonhl::AddressOfOp::create(
                  returnBuilder, operation.getLoc(),
                  ::vrc::udonhl::PtrType::get(module.getContext(), 0),
                  payloadSymbol.getValue());
              auto load = ::vrc::udonhl::LoadOp::create(
                  returnBuilder, operation.getLoc(), resultType,
                  address.getResult(),
                  frameAlignment(
                      physicalFrameType(resultType, module.getContext())));
              load->setAttr("vrc.async_public_result_load",
                            UnitAttr::get(module.getContext()));
              returnValues.push_back(load.getResult());
              if (physicalResults.empty())
                physicalResults.push_back(resultType);
              else if (physicalResults.size() != 1 ||
                       physicalResults.front() != resultType) {
                operation.emitError(
                    "public async export return types do not agree");
                invalid = true;
              }
            }
          }
        } else if (publicExport && operation.getNumOperands() > 1) {
          operation.emitError(
              "public async exports support at most one result");
          invalid = true;
        }
        func::ReturnOp::create(returnBuilder, operation.getLoc(), returnValues);
        operation.erase();
      }
      function.setType(FunctionType::get(
          module.getContext(), function.getArgumentTypes(), physicalResults));
      function->removeAttr("llvm.passthrough");
    }

    SmallVector<func::CallOp> calls;
    module.walk([&](func::CallOp call) {
      if (coroutineNames.count(call.getCallee().str()))
        calls.push_back(call);
    });
    for (func::CallOp call : calls) {
      if (llvm::any_of(call.getResults(),
                       [](Value result) { return !result.use_empty(); })) {
        call.emitError("prepared coroutine call still has live async results");
        invalid = true;
        continue;
      }
      OpBuilder callBuilder(call);
      auto replacement =
          func::CallOp::create(callBuilder, call.getLoc(), call.getCallee(),
                               TypeRange{}, call.getOperands());
      for (NamedAttribute attribute : call->getAttrs()) {
        StringRef name = attribute.getName().strref();
        if (name == "callee" || name == "vrc.async_result_object_ids")
          continue;
        replacement->setAttr(attribute.getName(), attribute.getValue());
      }
      if (Attribute ids = call->getAttr("vrc.async_result_object_ids"))
        replacement->setAttr("vrc.async_callee_result_object_ids", ids);
      call.erase();
    }
    if (invalid) {
      signalPassFailure();
      return;
    }

    for (func::FuncOp function : coroutines) {
      SmallVector<Operation *> eventObjects;
      SmallVector<Operation *> suspends;
      SmallVector<Operation *> scaffold;
      function.walk([&](Operation *operation) {
        StringRef name = operation->getName().getStringRef();
        if (name == "vrc.event_as_async")
          eventObjects.push_back(operation);
        else if (name == "async.coro.suspend")
          suspends.push_back(operation);
        if (name == "async.runtime.create" ||
            (name.starts_with("async.coro.") && name != "async.coro.suspend"))
          scaffold.push_back(operation);
      });

      for (Operation *eventObject : eventObjects) {
        auto ids = eventObject->getAttrOfType<DenseI32ArrayAttr>(
            "vrc.async_result_object_ids");
        if (!ids || ids.size() != 1 || ids[0] <= 0 ||
            eventObject->getNumOperands() != 1 ||
            eventObject->getNumResults() != 1) {
          eventObject->emitError(
              "event async object requires one canonical object id");
          invalid = true;
          continue;
        }
        if (!eventObject->getResult(0).use_empty()) {
          eventObject->emitError(
              "event async object still has uses after await lowering");
          invalid = true;
          continue;
        }
        OpBuilder bindBuilder(eventObject);
        OperationState bindState(
            eventObject->getLoc(),
            ::vrc::udonhl::AsyncBindEventOp::getOperationName());
        bindState.addOperands(eventObject->getOperand(0));
        bindState.addAttribute("object_id",
                               bindBuilder.getI32IntegerAttr(ids[0]));
        if (checkpointRoutes.count(ids[0]))
          bindState.addAttribute("vrc.cooperative_checkpoint",
                                 bindBuilder.getUnitAttr());
        bindBuilder.create(bindState);
        boundObjectIds.insert(ids[0]);
        eventObject->erase();
      }
      for (Operation *suspend : suspends) {
        if (suspend->getNumSuccessors() < 1) {
          suspend->emitError("coroutine suspend has no return successor");
          invalid = true;
          continue;
        }
        OpBuilder branchBuilder(suspend);
        cf::BranchOp::create(branchBuilder, suspend->getLoc(),
                             suspend->getSuccessor(0));
        suspend->erase();
      }

      for (Operation *operation : llvm::reverse(scaffold)) {
        if (!operation->getBlock())
          continue;
        if (llvm::any_of(operation->getResults(),
                         [](Value result) { return !result.use_empty(); })) {
          operation->emitError("coroutine scaffold still has live results");
          invalid = true;
          continue;
        }
        operation->erase();
      }
    }
    if (invalid) {
      signalPassFailure();
      return;
    }

    if (!boundObjectIds.empty()) {
      constexpr StringLiteral completionName = "__udoncl_async_complete";
      if (module.lookupSymbol(completionName)) {
        module.emitError("reserved async completion symbol already exists");
        signalPassFailure();
        return;
      }
      OpBuilder completionBuilder(module.getBodyRegion());
      completionBuilder.setInsertionPointToEnd(module.getBody());
      auto i32 = completionBuilder.getI32Type();
      auto completion = func::FuncOp::create(
          completionBuilder, module.getLoc(), completionName,
          FunctionType::get(module.getContext(), {i32, i32}, {}));
      completion.setPublic();
      completion->setAttr("vrc.compiler_async_completion",
                          completionBuilder.getUnitAttr());
      completion->setAttr("vrc.udon_event", completionBuilder.getStringAttr(
                                                "__udoncl_async_complete"));

      Block *entry = completion.addEntryBlock();
      SmallVector<Block *> checks;
      SmallVector<Block *> completions;
      for (size_t index = 0; index < boundObjectIds.size(); ++index) {
        checks.push_back(new Block());
        completions.push_back(new Block());
        completion.getBody().push_back(checks.back());
        completion.getBody().push_back(completions.back());
      }
      Block *unmatched = new Block();
      completion.getBody().push_back(unmatched);
      completionBuilder.setInsertionPointToEnd(entry);
      cf::BranchOp::create(completionBuilder, module.getLoc(), checks.front());

      auto addressOf = [&](OpBuilder &builder, int32_t objectId,
                           StringRef suffix) {
        std::string symbol = "__udoncl_async_object_" +
                             std::to_string(objectId) + "_" + suffix.str();
        return ::vrc::udonhl::AddressOfOp::create(
            builder, module.getLoc(),
            ::vrc::udonhl::PtrType::get(module.getContext(), 0), symbol);
      };
      auto loadI32 = [&](OpBuilder &builder,
                         ::vrc::udonhl::AddressOfOp address) {
        return ::vrc::udonhl::LoadOp::create(builder, module.getLoc(), i32,
                                             address.getResult(), 4);
      };
      auto storeI32 = [&](OpBuilder &builder, Value value,
                          ::vrc::udonhl::AddressOfOp address,
                          StringRef marker) {
        auto store = ::vrc::udonhl::StoreOp::create(
            builder, module.getLoc(), value, address.getResult(), 4);
        if (!marker.empty())
          store->setAttr(marker, builder.getUnitAttr());
      };

      size_t routeIndex = 0;
      for (int32_t objectId : boundObjectIds) {
        Block *check = checks[routeIndex];
        Block *complete = completions[routeIndex];
        Block *next =
            routeIndex + 1 < checks.size() ? checks[routeIndex + 1] : unmatched;
        completionBuilder.setInsertionPointToEnd(check);
        auto id = arith::ConstantIntOp::create(completionBuilder,
                                               module.getLoc(), objectId, 32);
        auto matches = arith::CmpIOp::create(completionBuilder, module.getLoc(),
                                             arith::CmpIPredicate::eq,
                                             entry->getArgument(0), id);
        cf::CondBranchOp::create(completionBuilder, module.getLoc(), matches,
                                 complete, ValueRange{}, next, ValueRange{});

        completionBuilder.setInsertionPointToEnd(complete);
        auto checkpoint = checkpointRoutes.find(objectId);
        if (checkpoint != checkpointRoutes.end()) {
          OperationState enterState(
              module.getLoc(), ::vrc::udonhl::AsyncEnterOp::getOperationName());
          enterState.addAttribute(
              "frame_id",
              completionBuilder.getI32IntegerAttr(checkpoint->second.first));
          enterState.addAttribute(
              "resume_pc",
              completionBuilder.getI32IntegerAttr(checkpoint->second.second));
          completionBuilder.create(enterState);
          func::ReturnOp::create(completionBuilder, module.getLoc());
          ++routeIndex;
          continue;
        }
        auto zero = arith::ConstantIntOp::create(completionBuilder,
                                                 module.getLoc(), 0, 32);
        auto available = arith::ConstantIntOp::create(completionBuilder,
                                                      module.getLoc(), 1, 32);
        auto error = arith::ConstantIntOp::create(completionBuilder,
                                                  module.getLoc(), -1, 32);
        auto succeeded = arith::CmpIOp::create(
            completionBuilder, module.getLoc(), arith::CmpIPredicate::eq,
            entry->getArgument(1), zero);
        auto state = arith::SelectOp::create(completionBuilder, module.getLoc(),
                                             succeeded, available, error);
        auto payloadAddress = addressOf(completionBuilder, objectId, "payload");
        auto stateAddress = addressOf(completionBuilder, objectId, "state");
        auto waiterFrameAddress =
            addressOf(completionBuilder, objectId, "waiter_frame");
        auto waiterPCAddress =
            addressOf(completionBuilder, objectId, "waiter_pc");
        storeI32(completionBuilder, entry->getArgument(1), payloadAddress,
                 "vrc.async_object_payload_store");
        storeI32(completionBuilder, state, stateAddress,
                 "vrc.async_object_completion_state");
        auto waiterFrame = loadI32(completionBuilder, waiterFrameAddress);
        waiterFrame->setAttr("vrc.async_object_waiter_frame_load",
                             completionBuilder.getUnitAttr());
        auto waiterPC = loadI32(completionBuilder, waiterPCAddress);
        waiterPC->setAttr("vrc.async_object_waiter_pc_load",
                          completionBuilder.getUnitAttr());
        storeI32(completionBuilder, zero, waiterFrameAddress,
                 "vrc.async_object_waiter_clear");
        storeI32(completionBuilder, zero, waiterPCAddress,
                 "vrc.async_object_waiter_clear");
        auto objectRoutes = objectResumeRoutes.find(objectId);
        if (objectRoutes != objectResumeRoutes.end() &&
            objectRoutes->second.size() == 1) {
          auto [frameId, resumePC] = *objectRoutes->second.begin();
          OperationState enterState(
              module.getLoc(), ::vrc::udonhl::AsyncEnterOp::getOperationName());
          enterState.addAttribute("frame_id",
                                  completionBuilder.getI32IntegerAttr(frameId));
          enterState.addAttribute(
              "resume_pc", completionBuilder.getI32IntegerAttr(resumePC));
          completionBuilder.create(enterState);
        } else if (objectRoutes != objectResumeRoutes.end() &&
                   !objectRoutes->second.empty()) {
          std::string dispatcher =
              "__udoncl_async_resume_object_" + std::to_string(objectId);
          func::CallOp::create(
              completionBuilder, module.getLoc(), dispatcher, TypeRange{},
              ValueRange{waiterFrame.getResult(), waiterPC.getResult()});
          specializedObjectDispatchers.insert(objectId);
        } else {
          OperationState resumeState(
              module.getLoc(),
              ::vrc::udonhl::AsyncResumeOp::getOperationName());
          resumeState.addOperands(
              {waiterFrame.getResult(), waiterPC.getResult()});
          completionBuilder.create(resumeState);
        }
        func::ReturnOp::create(completionBuilder, module.getLoc());
        ++routeIndex;
      }
      completionBuilder.setInsertionPointToEnd(unmatched);
      func::ReturnOp::create(completionBuilder, module.getLoc());
    }

    module.walk([&](::vrc::udonhl::AsyncResumeOp resume) {
      auto objectId = resume->getAttrOfType<IntegerAttr>("vrc.async_object_id");
      if (objectId && objectResumeRoutes.count(objectId.getInt()))
        specializedObjectDispatchers.insert(
            static_cast<int32_t>(objectId.getInt()));
    });
    for (int32_t objectId : specializedObjectDispatchers) {
      std::string name =
          "__udoncl_async_resume_object_" + std::to_string(objectId);
      if (module.lookupSymbol(name)) {
        module.emitError("reserved async object dispatcher already exists: ")
            << name;
        signalPassFailure();
        return;
      }
      const auto &objectRoutes = objectResumeRoutes.at(objectId);
      OpBuilder objectBuilder(module.getBodyRegion());
      objectBuilder.setInsertionPointToEnd(module.getBody());
      auto i32 = objectBuilder.getI32Type();
      auto dispatcher = func::FuncOp::create(
          objectBuilder, module.getLoc(), name,
          FunctionType::get(module.getContext(), {i32, i32}, {}));
      dispatcher.setPrivate();
      dispatcher->setAttr("vrc.compiler_async_object_dispatch",
                          objectBuilder.getI32IntegerAttr(objectId));
      Block *entry = dispatcher.addEntryBlock();
      SmallVector<Block *> checks;
      SmallVector<Block *> entries;
      for (size_t index = 0; index < objectRoutes.size(); ++index) {
        checks.push_back(new Block());
        entries.push_back(new Block());
        dispatcher.getBody().push_back(checks.back());
        dispatcher.getBody().push_back(entries.back());
      }
      Block *miss = new Block();
      dispatcher.getBody().push_back(miss);
      objectBuilder.setInsertionPointToEnd(entry);
      cf::BranchOp::create(objectBuilder, module.getLoc(), checks.front());
      size_t index = 0;
      for (auto [frameId, resumePC] : objectRoutes) {
        Block *next = index + 1 < checks.size() ? checks[index + 1] : miss;
        objectBuilder.setInsertionPointToEnd(checks[index]);
        auto frame = arith::ConstantIntOp::create(objectBuilder,
                                                  module.getLoc(), frameId, 32);
        auto pc = arith::ConstantIntOp::create(objectBuilder, module.getLoc(),
                                               resumePC, 32);
        auto frameMatches = arith::CmpIOp::create(
            objectBuilder, module.getLoc(), arith::CmpIPredicate::eq,
            entry->getArgument(0), frame);
        auto pcMatches = arith::CmpIOp::create(objectBuilder, module.getLoc(),
                                               arith::CmpIPredicate::eq,
                                               entry->getArgument(1), pc);
        auto matches = arith::AndIOp::create(objectBuilder, module.getLoc(),
                                             frameMatches, pcMatches);
        cf::CondBranchOp::create(objectBuilder, module.getLoc(), matches,
                                 entries[index], ValueRange{}, next,
                                 ValueRange{});
        objectBuilder.setInsertionPointToEnd(entries[index]);
        OperationState enterState(
            module.getLoc(), ::vrc::udonhl::AsyncEnterOp::getOperationName());
        enterState.addAttribute("frame_id",
                                objectBuilder.getI32IntegerAttr(frameId));
        enterState.addAttribute("resume_pc",
                                objectBuilder.getI32IntegerAttr(resumePC));
        objectBuilder.create(enterState);
        func::ReturnOp::create(objectBuilder, module.getLoc());
        ++index;
      }
      objectBuilder.setInsertionPointToEnd(miss);
      func::ReturnOp::create(objectBuilder, module.getLoc());
    }

    constexpr StringLiteral dispatcherName = "__udoncl_async_resume_dispatch";
    if (module.lookupSymbol(dispatcherName)) {
      module.emitError(
          "reserved async resume dispatcher symbol already exists");
      signalPassFailure();
      return;
    }
    OpBuilder dispatcherBuilder(module.getBodyRegion());
    dispatcherBuilder.setInsertionPointToEnd(module.getBody());
    auto i32 = dispatcherBuilder.getI32Type();
    auto dispatcher = func::FuncOp::create(
        dispatcherBuilder, module.getLoc(), dispatcherName,
        FunctionType::get(module.getContext(), {i32, i32}, {}));
    dispatcher.setPrivate();
    dispatcher->setAttr("vrc.compiler_async_dispatch",
                        dispatcherBuilder.getUnitAttr());
    Block *dispatcherEntry = dispatcher.addEntryBlock();
    SmallVector<Block *> routeChecks;
    SmallVector<Block *> routeEntries;
    for (size_t index = 0; index < routes.size(); ++index) {
      routeChecks.push_back(new Block());
      routeEntries.push_back(new Block());
      dispatcher.getBody().push_back(routeChecks.back());
      dispatcher.getBody().push_back(routeEntries.back());
    }
    Block *routeMiss = new Block();
    dispatcher.getBody().push_back(routeMiss);
    dispatcherBuilder.setInsertionPointToEnd(dispatcherEntry);
    if (routeChecks.empty())
      cf::BranchOp::create(dispatcherBuilder, module.getLoc(), routeMiss);
    else
      cf::BranchOp::create(dispatcherBuilder, module.getLoc(),
                           routeChecks.front());
    for (auto [index, attribute] : llvm::enumerate(routes)) {
      auto route = cast<DictionaryAttr>(attribute);
      int64_t frameId = route.getAs<IntegerAttr>("frame_id").getInt();
      int64_t resumePC = route.getAs<IntegerAttr>("resume_pc").getInt();
      Block *next =
          index + 1 < routeChecks.size() ? routeChecks[index + 1] : routeMiss;
      dispatcherBuilder.setInsertionPointToEnd(routeChecks[index]);
      auto frame = arith::ConstantIntOp::create(dispatcherBuilder,
                                                module.getLoc(), frameId, 32);
      auto pc = arith::ConstantIntOp::create(dispatcherBuilder, module.getLoc(),
                                             resumePC, 32);
      auto frameMatches = arith::CmpIOp::create(
          dispatcherBuilder, module.getLoc(), arith::CmpIPredicate::eq,
          dispatcherEntry->getArgument(0), frame);
      auto pcMatches = arith::CmpIOp::create(
          dispatcherBuilder, module.getLoc(), arith::CmpIPredicate::eq,
          dispatcherEntry->getArgument(1), pc);
      auto matches = arith::AndIOp::create(dispatcherBuilder, module.getLoc(),
                                           frameMatches, pcMatches);
      cf::CondBranchOp::create(dispatcherBuilder, module.getLoc(), matches,
                               routeEntries[index], ValueRange{}, next,
                               ValueRange{});
      dispatcherBuilder.setInsertionPointToEnd(routeEntries[index]);
      OperationState enterState(
          module.getLoc(), ::vrc::udonhl::AsyncEnterOp::getOperationName());
      enterState.addAttribute("frame_id",
                              dispatcherBuilder.getI32IntegerAttr(frameId));
      enterState.addAttribute("resume_pc",
                              dispatcherBuilder.getI32IntegerAttr(resumePC));
      dispatcherBuilder.create(enterState);
      func::ReturnOp::create(dispatcherBuilder, module.getLoc());
    }
    dispatcherBuilder.setInsertionPointToEnd(routeMiss);
    func::ReturnOp::create(dispatcherBuilder, module.getLoc());

    SmallVector<::vrc::udonhl::AsyncResumeOp> resumeRequests;
    module.walk([&](::vrc::udonhl::AsyncResumeOp resume) {
      resumeRequests.push_back(resume);
    });
    for (::vrc::udonhl::AsyncResumeOp resume : resumeRequests) {
      OpBuilder callBuilder(resume);
      auto objectId = resume->getAttrOfType<IntegerAttr>("vrc.async_object_id");
      std::string target =
          objectId && specializedObjectDispatchers.count(objectId.getInt())
              ? "__udoncl_async_resume_object_" +
                    std::to_string(objectId.getInt())
              : dispatcherName.str();
      func::CallOp::create(callBuilder, resume.getLoc(), target, TypeRange{},
                           resume.getOperands());
      resume.erase();
    }

    module.walk([&](Operation *operation) {
      if (operation->getName().getDialectNamespace() == "async") {
        operation->emitError("upstream Async operation remains after Udon "
                             "state-machine lowering");
        invalid = true;
      }
    });
    if (invalid) {
      signalPassFailure();
      return;
    }
    // Suspension is fully represented by static frames, async objects, and
    // resume routes from this point onward. Keeping the source effect on a
    // lowered coroutine would make the later func.call -> udonhl.call
    // conversion require source-level propagation a second time, including
    // through compiler-generated completion/dispatch helpers.
    for (func::FuncOp function : coroutines)
      function->removeAttr("vrc.async_effect");
    module->setAttr("vrc.async_stage",
                    StringAttr::get(module.getContext(), "udon-static-v1"));
  }
};
} // namespace
} // namespace vrc
