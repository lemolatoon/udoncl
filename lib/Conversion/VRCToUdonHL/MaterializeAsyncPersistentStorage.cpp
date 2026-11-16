#include "VRC/Conversion/Passes.h"
#include "VRC/Analysis/HostStorageLiveness.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "VRC/Support/HostStorageType.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include <map>
#include <string>

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_MATERIALIZEASYNCPERSISTENTSTORAGEPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {

struct PersistentMaterialization {
  StringAttr requestId;
  StringAttr physicalType;
  IntegerAttr priority;
  SmallVector<StringAttr> conflicts;
  Value definition;
};

struct MaterializeAsyncPersistentStoragePass
    : impl::MaterializeAsyncPersistentStoragePassBase<
          MaterializeAsyncPersistentStoragePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto version = module->getAttrOfType<IntegerAttr>(
        "vrc.async_persistent_request_version");
    auto requests =
        module->getAttrOfType<ArrayAttr>("vrc.async_persistent_requests");
    if (!version && !requests)
      return;
    if (!version || version.getInt() != 1 || !requests) {
      module.emitError("invalid async persistent materialization schema");
      signalPassFailure();
      return;
    }

    SmallVector<func::FuncOp> functions;
    for (func::FuncOp function : module.getOps<func::FuncOp>())
      if (!function.isExternal())
        functions.push_back(function);
    llvm::sort(functions, [](func::FuncOp left, func::FuncOp right) {
      return left.getSymName() < right.getSymName();
    });
    llvm::DenseSet<Operation *> constants;
    auto analysis = analyzeHostStorageInterference(
        module, functions,
        [](Type type) -> FailureOr<std::string> {
          return hostStorageKey(type);
        },
        constants, true);
    if (failed(analysis)) {
      signalPassFailure();
      return;
    }
    std::map<std::string, Value> valueById;
    for (const auto &[value, id] : analysis->valueIds)
      valueById.emplace(id, value);

    Builder builder(module.getContext());
    SmallVector<PersistentMaterialization> normalizedRequests;
    SmallVector<Operation *> erase;
    DenseSet<Value> claimedMembers;
    for (Attribute attribute : requests) {
      auto request = dyn_cast<DictionaryAttr>(attribute);
      auto requestId =
          request ? request.getAs<StringAttr>("request_id") : StringAttr();
      auto priority =
          request ? request.getAs<IntegerAttr>("priority") : IntegerAttr();
      auto physicalType =
          request ? request.getAs<StringAttr>("physical_type") : StringAttr();
      auto conflicts =
          request ? request.getAs<ArrayAttr>("interferes_with") : ArrayAttr();
      auto members =
          request ? request.getAs<ArrayAttr>("member_value_ids") : ArrayAttr();
      if (!request || !requestId || requestId.getValue().empty() || !priority ||
          !priority.getType().isInteger(64) || priority.getInt() < 1 ||
          !physicalType || physicalType.getValue().empty() || !conflicts ||
          !members || members.empty()) {
        module.emitError("invalid async persistent materialization request");
        signalPassFailure();
        return;
      }
      DenseSet<Value> memberValues;
      for (Attribute memberAttribute : members) {
        auto memberId = dyn_cast<StringAttr>(memberAttribute);
        auto member = memberId ? valueById.find(memberId.getValue().str())
                               : valueById.end();
        if (!memberId || member == valueById.end() ||
            !memberValues.insert(member->second).second ||
            !claimedMembers.insert(member->second).second) {
          module.emitError("invalid async persistent materialization member");
          signalPassFailure();
          return;
        }
      }

      udonhl::SpillStoreOp definitionStore;
      bool duplicateDefinition = false;
      SmallVector<udonhl::SpillLoadOp> reloads;
      module.walk([&](Operation *operation) {
        auto id = operation->getAttrOfType<StringAttr>(
            "vrc.async_persistent_request_id");
        if (!id || id != requestId)
          return;
        if (auto store = dyn_cast<udonhl::SpillStoreOp>(operation)) {
          duplicateDefinition |= bool(definitionStore);
          definitionStore = store;
        } else if (auto load = dyn_cast<udonhl::SpillLoadOp>(operation)) {
          reloads.push_back(load);
        }
      });
      if (duplicateDefinition || !definitionStore || reloads.empty() ||
          !memberValues.contains(definitionStore.getValue())) {
        module.emitError("async persistent request lacks unique frame aliases");
        signalPassFailure();
        return;
      }
      for (udonhl::SpillLoadOp reload : reloads) {
        if (!memberValues.contains(reload.getValue())) {
          reload.emitError("async persistent reload is outside its request");
          signalPassFailure();
          return;
        }
        reload.getValue().replaceAllUsesWith(definitionStore.getValue());
        erase.push_back(reload);
      }
      erase.push_back(definitionStore);
      SmallVector<StringAttr> conflictIds;
      for (Attribute conflict : conflicts) {
        auto conflictId = dyn_cast<StringAttr>(conflict);
        if (!conflictId || conflictId.getValue().empty()) {
          module.emitError("invalid async persistent conflict identity");
          signalPassFailure();
          return;
        }
        conflictIds.push_back(conflictId);
      }
      normalizedRequests.push_back({requestId, physicalType, priority,
                                    std::move(conflictIds),
                                    definitionStore.getValue()});
    }
    for (Operation *operation : erase)
      operation->erase();
    for (func::FuncOp function : functions) {
      function->removeAttr("vrc.async_spill_slots");
      function->removeAttr("vrc.async_spill_slot_count");
      function.walk([](Operation *operation) {
        operation->removeAttr("vrc.async_live_spill_slots");
      });
    }
    module->removeAttr("vrc.async_persistent_request_version");
    module->removeAttr("vrc.async_persistent_requests");

    auto normalizedAnalysis = analyzeHostStorageInterference(
        module, functions,
        [](Type type) -> FailureOr<std::string> {
          return hostStorageKey(type);
        },
        constants, true);
    if (failed(normalizedAnalysis)) {
      signalPassFailure();
      return;
    }
    std::map<std::string, std::string> valueIdByRequest;
    for (const PersistentMaterialization &request : normalizedRequests) {
      auto valueId = normalizedAnalysis->valueIds.find(request.definition);
      auto storageKey =
          normalizedAnalysis->storageKeys.find(request.definition);
      if (valueId == normalizedAnalysis->valueIds.end() ||
          storageKey == normalizedAnalysis->storageKeys.end() ||
          storageKey->second != request.physicalType.getValue()) {
        module.emitError("normalized async persistent value lacks identity");
        signalPassFailure();
        return;
      }
      valueIdByRequest.emplace(request.requestId.getValue().str(),
                               valueId->second);
    }
    SmallVector<Attribute> materialized;
    for (const PersistentMaterialization &request : normalizedRequests) {
      SmallVector<Attribute> canonicalConflicts;
      for (StringAttr conflict : request.conflicts) {
        auto found = valueIdByRequest.find(conflict.getValue().str());
        if (found == valueIdByRequest.end()) {
          module.emitError("async persistent conflict lacks canonical value");
          signalPassFailure();
          return;
        }
        canonicalConflicts.push_back(builder.getStringAttr(found->second));
      }
      materialized.push_back(builder.getDictionaryAttr({
          builder.getNamedAttr("request_id", request.requestId),
          builder.getNamedAttr("canonical_value_id",
                               builder.getStringAttr(valueIdByRequest.at(
                                   request.requestId.getValue().str()))),
          builder.getNamedAttr("physical_type", request.physicalType),
          builder.getNamedAttr("priority", request.priority),
          builder.getNamedAttr("interferes_with",
                               builder.getArrayAttr(canonicalConflicts)),
      }));
    }
    module->setAttr("vrc.async_persistent_materialization_version",
                    builder.getI32IntegerAttr(2));
    module->setAttr("vrc.async_persistent_direct_requests",
                    builder.getArrayAttr(materialized));
  }
};

} // namespace
} // namespace vrc
