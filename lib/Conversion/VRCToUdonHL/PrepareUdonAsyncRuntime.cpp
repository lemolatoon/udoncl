#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/VRC/VRCDialect.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_PREPAREUDONASYNCRUNTIMEPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {
namespace runtime = ::vrc::vrc;

bool isAsyncObjectType(Type type) {
  return isa<async::ValueType, async::TokenType>(type);
}

bool isRecomputableAcrossSuspend(Value value) {
  auto result = dyn_cast<OpResult>(value);
  if (!result)
    return false;
  StringRef name = result.getOwner()->getName().getStringRef();
  // Scalar lowering gives constants immutable Udon heap slots.  address_of is
  // conceptually recomputable too, but this pass does not clone it after each
  // resume; it must therefore be demoted like every other live pointer.
  return name == "arith.constant";
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

std::string physicalFrameStorageKey(Type type, MLIRContext *context) {
  type = physicalFrameType(type, context);
  if (auto integer = dyn_cast<IntegerType>(type)) {
    if (integer.getWidth() == 1)
      return "SystemBoolean";
    return integer.getWidth() > 32 ? "SystemInt64" : "SystemInt32";
  }
  if (type.isF32())
    return "SystemSingle";
  if (isa<::vrc::udonhl::PtrType>(type))
    return "SystemInt32";
  return {};
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
    std::map<std::string, int32_t> nextPersistentSpillIndex;
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
        DenseMap<Value, llvm::DenseSet<Value>> interference;
        if (!annotateResultIds) {
          llvm::DenseSet<Value> planned;
          for (const PlannedValue &value : values)
            planned.insert(value.value);
          function.walk([&](Operation *operation) {
            const LivenessBlockInfo *blockInfo =
                liveness.getLiveness(operation->getBlock());
            if (!blockInfo)
              return;
            SmallVector<Value> simultaneous;
            for (Value value : blockInfo->currentlyLiveValues(operation))
              if (planned.contains(value))
                simultaneous.push_back(value);
            for (Value operand : operation->getOperands())
              if (planned.contains(operand))
                simultaneous.push_back(operand);
            for (Value result : operation->getResults())
              if (planned.contains(result))
                simultaneous.push_back(result);
            llvm::sort(simultaneous, [&](Value left, Value right) {
              return valueIds.lookup(left) < valueIds.lookup(right);
            });
            simultaneous.erase(
                std::unique(simultaneous.begin(), simultaneous.end()),
                simultaneous.end());
            for (auto [leftIndex, left] : llvm::enumerate(simultaneous)) {
              std::string leftKey =
                  physicalFrameStorageKey(left.getType(), module.getContext());
              for (Value right :
                   ArrayRef<Value>(simultaneous).drop_front(leftIndex + 1)) {
                if (leftKey != physicalFrameStorageKey(right.getType(),
                                                       module.getContext()))
                  continue;
                interference[left].insert(right);
                interference[right].insert(left);
              }
            }
          });
        }
        int32_t slotCount = 0;
        if (!annotateResultIds) {
          // A live-across-suspend SSA value owns one stable frame slot for
          // the entire coroutine invocation.  Sharing slots merely because
          // resume-PC sets do not overlap is insufficient after full
          // demotion: definitions can execute on fast/error paths that do not
          // suspend, and a later definition must never overwrite a value
          // still selected by a control-flow merge.
          slotCount = static_cast<int32_t>(values.size());
          for (PlannedValue &value : values) {
            std::string storageKey = physicalFrameStorageKey(
                value.value.getType(), module.getContext());
            if (storageKey.empty()) {
              function.emitError("cannot assign async persistent storage for ")
                  << value.value.getType();
              invalid = true;
              continue;
            }
            value.slot = nextPersistentSpillIndex[storageKey]++;
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
          if (!annotateResultIds) {
            attributes.append("storage_key",
                              builder.getStringAttr(physicalFrameStorageKey(
                                  value.value.getType(), module.getContext())));
            attributes.append("placement",
                              builder.getStringAttr("typed_spill_array"));
            SmallVector<int32_t> conflicts;
            for (Value conflict : interference[value.value])
              conflicts.push_back(valueIds.lookup(conflict));
            llvm::sort(conflicts);
            attributes.append("interferes_with",
                              builder.getDenseI32ArrayAttr(conflicts));
          } else {
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
          for (const PlannedValue &value : values) {
            SmallVector<OpOperand *> uses;
            for (OpOperand &use : value.value.getUses())
              uses.push_back(&use);

            // Store at the SSA definition, rather than immediately before a
            // suspend.  Every path on which the value exists therefore
            // initializes the frame before any fast-path, resume-path, or
            // error-path merge can consume it.
            OpBuilder storeBuilder(module.getContext());
            if (auto argument = dyn_cast<BlockArgument>(value.value)) {
              storeBuilder.setInsertionPointToStart(argument.getOwner());
            } else {
              Operation *definition = value.value.getDefiningOp();
              if (!definition ||
                  definition->hasTrait<OpTrait::IsTerminator>()) {
                function.emitError(
                    "live-across-suspend value has no storable definition");
                invalid = true;
                break;
              }
              storeBuilder.setInsertionPointAfter(definition);
            }
            std::string storageKey = physicalFrameStorageKey(
                value.value.getType(), module.getContext());
            auto store = ::vrc::udonhl::SpillStoreOp::create(
                storeBuilder, value.value.getLoc(), value.value,
                builder.getStringAttr(storageKey),
                builder.getI32IntegerAttr(value.slot));
            store->setAttr("vrc.async_frame_store", builder.getUnitAttr());

            // Reload at every original use.  This is a real demotion to the
            // coroutine frame, so correctness does not depend on one resume
            // block dominating all later merge points.  Reuse one load when
            // an operation consumes the value more than once.
            DenseMap<Operation *, Value> loads;
            for (OpOperand *use : uses) {
              Operation *owner = use->getOwner();
              auto existing = loads.find(owner);
              Value loaded;
              if (existing != loads.end()) {
                loaded = existing->second;
              } else {
                OpBuilder loadBuilder(owner);
                auto load = ::vrc::udonhl::SpillLoadOp::create(
                    loadBuilder, owner->getLoc(), value.value.getType(),
                    builder.getStringAttr(storageKey),
                    builder.getI32IntegerAttr(value.slot));
                load->setAttr("vrc.async_frame_load", builder.getUnitAttr());
                loaded = load.getResult();
                loads.try_emplace(owner, loaded);
              }
              use->set(loaded);
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

} // namespace
} // namespace vrc
