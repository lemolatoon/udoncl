#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/VRC/VRCDialect.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
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

Type physicalFrameType(Type type, MLIRContext *context) {
  if (isa<runtime::BufferType, runtime::EventType, runtime::DispatchDescType>(
          type))
    return IntegerType::get(context, 32);
  return type;
}

uint64_t frameAlignment(Type type) {
  if (auto integer = dyn_cast<IntegerType>(type))
    return integer.getWidth() <= 8    ? 1
           : integer.getWidth() <= 16 ? 2
           : integer.getWidth() <= 32 ? 4
                                      : 8;
  return 4;
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

    // Udon functions return through static per-function (or per-bank) return
    // slots, so a continuation must never be entered while another activation
    // of the dispatcher or of the resumed coroutine is still waiting to
    // return. Entering the waiter directly from the completing child nests
    // dispatcher -> coroutine -> dispatcher; once a resumed parent calls the
    // same child again, the outer return slot is overwritten and the event
    // loops forever. A completing coroutine therefore only records its single
    // ready waiter and returns. The completion event drains that one-element
    // trampoline after every entry, so each continuation runs at dispatcher
    // depth one. The single-waiter/single-root contract guarantees that at
    // most one waiter is ready between drains; a zero frame is a no-op.
    SmallVector<::vrc::udonhl::AsyncResumeOp> resumeRequests;
    module.walk([&](::vrc::udonhl::AsyncResumeOp resume) {
      resumeRequests.push_back(resume);
    });
    if (!resumeRequests.empty()) {
      constexpr StringLiteral readyFrameName = "__udoncl_async_ready_frame";
      constexpr StringLiteral readyPCName = "__udoncl_async_ready_pc";
      constexpr StringLiteral drainName = "__udoncl_async_drain_ready";
      for (StringRef reserved : {StringRef(readyFrameName),
                                 StringRef(readyPCName), StringRef(drainName)})
        if (module.lookupSymbol(reserved)) {
          module.emitError("reserved async trampoline symbol already exists: ")
              << reserved;
          signalPassFailure();
          return;
        }
      OpBuilder trampolineBuilder(module.getBodyRegion());
      trampolineBuilder.setInsertionPointToEnd(module.getBody());
      auto i32 = trampolineBuilder.getI32Type();
      for (StringRef symbol :
           {StringRef(readyFrameName), StringRef(readyPCName)})
        ::vrc::udonhl::GlobalOp::create(
            trampolineBuilder, module.getLoc(), symbol, i32,
            trampolineBuilder.getI32IntegerAttr(0), "internal", 0, 4, false);
      auto readyAddress = [&](OpBuilder &builder, Location location,
                              StringRef symbol) {
        return ::vrc::udonhl::AddressOfOp::create(
            builder, location,
            ::vrc::udonhl::PtrType::get(module.getContext(), 0), symbol);
      };
      for (::vrc::udonhl::AsyncResumeOp resume : resumeRequests) {
        OpBuilder deferBuilder(resume);
        auto frameStore = ::vrc::udonhl::StoreOp::create(
            deferBuilder, resume.getLoc(), resume.getFrameId(),
            readyAddress(deferBuilder, resume.getLoc(), readyFrameName)
                .getResult(),
            4);
        frameStore->setAttr("vrc.async_ready_frame_store",
                            deferBuilder.getUnitAttr());
        auto pcStore = ::vrc::udonhl::StoreOp::create(
            deferBuilder, resume.getLoc(), resume.getResumePc(),
            readyAddress(deferBuilder, resume.getLoc(), readyPCName)
                .getResult(),
            4);
        pcStore->setAttr("vrc.async_ready_pc_store",
                         deferBuilder.getUnitAttr());
        resume.erase();
      }

      auto drain =
          func::FuncOp::create(trampolineBuilder, module.getLoc(), drainName,
                               FunctionType::get(module.getContext(), {}, {}));
      drain.setPrivate();
      drain->setAttr("vrc.compiler_async_drain",
                     trampolineBuilder.getUnitAttr());
      Block *drainEntry = drain.addEntryBlock();
      Block *check = new Block();
      Block *run = new Block();
      Block *done = new Block();
      drain.getBody().push_back(check);
      drain.getBody().push_back(run);
      drain.getBody().push_back(done);
      OpBuilder drainBuilder = OpBuilder::atBlockEnd(drainEntry);
      cf::BranchOp::create(drainBuilder, module.getLoc(), check);
      drainBuilder.setInsertionPointToEnd(check);
      auto frame = ::vrc::udonhl::LoadOp::create(
          drainBuilder, module.getLoc(), i32,
          readyAddress(drainBuilder, module.getLoc(), readyFrameName)
              .getResult(),
          4);
      auto zero =
          arith::ConstantIntOp::create(drainBuilder, module.getLoc(), 0, 32);
      auto ready = arith::CmpIOp::create(drainBuilder, module.getLoc(),
                                         arith::CmpIPredicate::ne,
                                         frame.getResult(), zero);
      cf::CondBranchOp::create(drainBuilder, module.getLoc(), ready, run,
                               ValueRange{}, done, ValueRange{});
      drainBuilder.setInsertionPointToEnd(run);
      auto runFrame = ::vrc::udonhl::LoadOp::create(
          drainBuilder, module.getLoc(), i32,
          readyAddress(drainBuilder, module.getLoc(), readyFrameName)
              .getResult(),
          4);
      auto runPC = ::vrc::udonhl::LoadOp::create(
          drainBuilder, module.getLoc(), i32,
          readyAddress(drainBuilder, module.getLoc(), readyPCName).getResult(),
          4);
      auto runZero =
          arith::ConstantIntOp::create(drainBuilder, module.getLoc(), 0, 32);
      ::vrc::udonhl::StoreOp::create(
          drainBuilder, module.getLoc(), runZero,
          readyAddress(drainBuilder, module.getLoc(), readyFrameName)
              .getResult(),
          4);
      ::vrc::udonhl::StoreOp::create(
          drainBuilder, module.getLoc(), runZero,
          readyAddress(drainBuilder, module.getLoc(), readyPCName).getResult(),
          4);
      func::CallOp::create(drainBuilder, module.getLoc(), dispatcherName,
                           TypeRange{},
                           ValueRange{runFrame.getResult(), runPC.getResult()});
      cf::BranchOp::create(drainBuilder, module.getLoc(), check);
      drainBuilder.setInsertionPointToEnd(done);
      func::ReturnOp::create(drainBuilder, module.getLoc());

      auto completion =
          module.lookupSymbol<func::FuncOp>("__udoncl_async_complete");
      if (completion && completion->hasAttr("vrc.compiler_async_completion")) {
        SmallVector<func::ReturnOp> completionReturns;
        completion.walk([&](func::ReturnOp operation) {
          completionReturns.push_back(operation);
        });
        for (func::ReturnOp operation : completionReturns) {
          OpBuilder drainCallBuilder(operation);
          func::CallOp::create(drainCallBuilder, operation.getLoc(), drainName,
                               TypeRange{}, ValueRange{});
        }
      }
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
