#include "VRC/Conversion/Passes.h"
#include "VRC/Analysis/HostStorageLiveness.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "VRC/Support/HostStorageType.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include <map>
#include <set>
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
  udonhl::SpillStoreOp store;
  SmallVector<udonhl::SpillLoadOp> reloads;
};

bool hasExactFields(DictionaryAttr dictionary,
                    ArrayRef<StringRef> expectedFields) {
  if (!dictionary || dictionary.size() != expectedFields.size())
    return false;
  return llvm::all_of(expectedFields, [&](StringRef field) {
    return bool(dictionary.get(field));
  });
}

LogicalResult validateCompletionReceipt(ModuleOp module, IntegerAttr version,
                                        ArrayAttr requests) {
  if (!version || version.getInt() != 2 || !requests)
    return module.emitError("invalid async persistent completion receipt");
  SmallVector<func::FuncOp> functions;
  for (func::FuncOp function : module.getOps<func::FuncOp>())
    if (!function.isExternal())
      functions.push_back(function);
  llvm::DenseSet<Operation *> constants;
  auto analysis = analyzeHostStorageInterference(
      module, functions,
      [](Type type) -> FailureOr<std::string> { return hostStorageKey(type); },
      constants, true);
  if (failed(analysis))
    return failure();
  std::map<std::string, Value> valuesById;
  for (const auto &[value, id] : analysis->valueIds)
    valuesById.emplace(id, value);
  std::set<std::string> requestIds;
  std::set<std::string> canonicalIds;
  std::map<std::string, std::set<std::string>> conflictsByValue;
  for (Attribute attribute : requests) {
    auto request = dyn_cast<DictionaryAttr>(attribute);
    if (!hasExactFields(request,
                        {"request_id", "canonical_value_id", "physical_type",
                         "priority", "interferes_with"}))
      return module.emitError("invalid async persistent completion request");
    auto requestId = request.getAs<StringAttr>("request_id");
    auto canonicalId = request.getAs<StringAttr>("canonical_value_id");
    auto physicalType = request.getAs<StringAttr>("physical_type");
    auto priority = request.getAs<IntegerAttr>("priority");
    auto conflicts = request.getAs<ArrayAttr>("interferes_with");
    auto value = canonicalId ? valuesById.find(canonicalId.getValue().str())
                             : valuesById.end();
    if (!requestId || requestId.getValue().empty() || !canonicalId ||
        canonicalId.getValue().empty() || !physicalType ||
        physicalType.getValue().empty() || !priority ||
        !priority.getType().isInteger(64) || priority.getInt() < 1 ||
        !conflicts || value == valuesById.end() ||
        analysis->storageKeys.lookup(value->second) !=
            physicalType.getValue() ||
        !requestIds.insert(requestId.getValue().str()).second ||
        !canonicalIds.insert(canonicalId.getValue().str()).second)
      return module.emitError("invalid async persistent completion binding");
    std::set<std::string> &conflictSet =
        conflictsByValue[canonicalId.getValue().str()];
    for (Attribute conflict : conflicts) {
      auto conflictId = dyn_cast<StringAttr>(conflict);
      if (!conflictId || conflictId.getValue().empty() ||
          conflictId == canonicalId ||
          !conflictSet.insert(conflictId.getValue().str()).second)
        return module.emitError("invalid async persistent completion conflict");
    }
  }
  for (const auto &[valueId, conflicts] : conflictsByValue)
    for (const std::string &conflict : conflicts) {
      auto reverse = conflictsByValue.find(conflict);
      if (!canonicalIds.count(conflict) || reverse == conflictsByValue.end() ||
          !reverse->second.count(valueId))
        return module.emitError(
            "asymmetric async persistent completion conflict");
    }
  bool hasLegacyState = false;
  module.walk([&](Operation *operation) {
    hasLegacyState |= operation->hasAttr("vrc.async_persistent_request_id") ||
                      operation->hasAttr("vrc.async_live_spill_slots") ||
                      isa<udonhl::SpillStoreOp, udonhl::SpillLoadOp>(operation);
  });
  for (func::FuncOp function : module.getOps<func::FuncOp>())
    hasLegacyState |= function->hasAttr("vrc.async_spill_slots") ||
                      function->hasAttr("vrc.async_spill_slot_count");
  if (hasLegacyState)
    return module.emitError(
        "completed async persistent storage retains legacy state");
  return success();
}

LogicalResult materializeAsyncPersistentStorage(ModuleOp module) {
  auto completionVersion = module->getAttrOfType<IntegerAttr>(
      "vrc.async_persistent_materialization_version");
  auto completionRequests =
      module->getAttrOfType<ArrayAttr>("vrc.async_persistent_direct_requests");
  auto version = module->getAttrOfType<IntegerAttr>(
      "vrc.async_persistent_request_version");
  auto requests =
      module->getAttrOfType<ArrayAttr>("vrc.async_persistent_requests");
  if (!version && !requests) {
    if (!completionVersion && !completionRequests)
      return success();
    return validateCompletionReceipt(module, completionVersion,
                                     completionRequests);
  }
  if (completionVersion || completionRequests)
    return module.emitError(
        "async persistent input already has a completion receipt");
  if (!version || version.getInt() != 1 || !requests) {
    module.emitError("invalid async persistent materialization schema");
    return failure();
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
      [](Type type) -> FailureOr<std::string> { return hostStorageKey(type); },
      constants, true);
  if (failed(analysis)) {
    return failure();
  }
  std::map<std::string, Value> valueById;
  for (const auto &[value, id] : analysis->valueIds)
    valueById.emplace(id, value);

  Builder builder(module.getContext());
  SmallVector<PersistentMaterialization> normalizedRequests;
  DenseSet<Value> claimedMembers;
  DenseSet<Operation *> claimedAliases;
  size_t markedAliasCount = 0;
  bool invalidMarker = false;
  module.walk([&](Operation *operation) {
    bool storeMarker = operation->hasAttr("vrc.async_frame_store");
    bool loadMarker = operation->hasAttr("vrc.async_frame_load");
    bool tagged = operation->hasAttr("vrc.async_persistent_request_id");
    if (storeMarker || loadMarker || tagged) {
      ++markedAliasCount;
      invalidMarker |= storeMarker == loadMarker || !tagged ||
                       (storeMarker && !isa<udonhl::SpillStoreOp>(operation)) ||
                       (loadMarker && !isa<udonhl::SpillLoadOp>(operation));
    }
  });
  if (invalidMarker)
    return module.emitError("invalid marked async persistent frame access");
  DominanceInfo dominance(module);
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
    auto alternatives =
        request ? request.getAs<ArrayAttr>("legal_alternatives") : ArrayAttr();
    auto compatibilityIndex =
        request ? request.getAs<IntegerAttr>("compatibility_index")
                : IntegerAttr();
    auto directAlternative = alternatives && alternatives.size() == 1
                                 ? dyn_cast<StringAttr>(alternatives[0])
                                 : StringAttr();
    if (!hasExactFields(request, {"request_id", "physical_type", "priority",
                                  "legal_alternatives", "member_value_ids",
                                  "interferes_with", "compatibility_index"}) ||
        !requestId || requestId.getValue().empty() || !priority ||
        !priority.getType().isInteger(64) || priority.getInt() < 1 ||
        !physicalType || physicalType.getValue().empty() || !conflicts ||
        !members || members.empty() || !alternatives || !directAlternative ||
        directAlternative.getValue() != "direct" || !compatibilityIndex ||
        !compatibilityIndex.getType().isInteger(32) ||
        compatibilityIndex.getInt() < 0) {
      module.emitError("invalid async persistent materialization request");
      return failure();
    }
    DenseSet<Value> memberValues;
    for (Attribute memberAttribute : members) {
      auto memberId = dyn_cast<StringAttr>(memberAttribute);
      auto member = memberId ? valueById.find(memberId.getValue().str())
                             : valueById.end();
      auto memberType = member != valueById.end()
                            ? hostStorageKey(member->second.getType())
                            : FailureOr<std::string>(failure());
      if (!memberId || member == valueById.end() || failed(memberType) ||
          *memberType != physicalType.getValue() ||
          !memberValues.insert(member->second).second ||
          !claimedMembers.insert(member->second).second) {
        module.emitError("invalid async persistent materialization member");
        return failure();
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
      claimedAliases.insert(operation);
      auto storageKey = operation->getAttrOfType<StringAttr>("storage_key");
      auto index = operation->getAttrOfType<IntegerAttr>("index");
      if (!storageKey || storageKey != physicalType || !index ||
          !index.getType().isInteger(32) ||
          index.getInt() != compatibilityIndex.getInt()) {
        duplicateDefinition = true;
        return;
      }
      if (auto store = dyn_cast<udonhl::SpillStoreOp>(operation)) {
        duplicateDefinition |= bool(definitionStore);
        definitionStore = store;
      } else if (auto load = dyn_cast<udonhl::SpillLoadOp>(operation)) {
        reloads.push_back(load);
      }
    });
    if (duplicateDefinition || !definitionStore || reloads.empty() ||
        !memberValues.contains(definitionStore.getValue()) ||
        memberValues.size() != reloads.size() + 1) {
      module.emitError("async persistent request lacks unique frame aliases");
      return failure();
    }
    for (udonhl::SpillLoadOp reload : reloads) {
      if (!memberValues.contains(reload.getValue())) {
        reload.emitError("async persistent reload is outside its request");
        return failure();
      }
      for (OpOperand &use : reload.getValue().getUses())
        if (!dominance.dominates(definitionStore.getValue(), use.getOwner()))
          return reload.emitError(
              "async persistent definition does not dominate reload use");
    }
    SmallVector<StringAttr> conflictIds;
    for (Attribute conflict : conflicts) {
      auto conflictId = dyn_cast<StringAttr>(conflict);
      if (!conflictId || conflictId.getValue().empty()) {
        module.emitError("invalid async persistent conflict identity");
        return failure();
      }
      conflictIds.push_back(conflictId);
    }
    normalizedRequests.push_back(
        {requestId, physicalType, priority, std::move(conflictIds),
         definitionStore.getValue(), definitionStore, std::move(reloads)});
  }
  if (claimedAliases.size() != markedAliasCount)
    return module.emitError(
        "async persistent materialization has an unknown marked operation");
  std::set<std::string> knownRequests;
  for (const PersistentMaterialization &request : normalizedRequests)
    if (!knownRequests.insert(request.requestId.getValue().str()).second)
      return module.emitError(
          "duplicate async persistent materialization request");
  std::map<std::string, std::set<std::string>> conflictsByRequest;
  for (const PersistentMaterialization &request : normalizedRequests) {
    std::set<std::string> &uniqueConflicts =
        conflictsByRequest[request.requestId.getValue().str()];
    for (StringAttr conflict : request.conflicts)
      if (conflict == request.requestId ||
          !knownRequests.count(conflict.getValue().str()) ||
          !uniqueConflicts.insert(conflict.getValue().str()).second)
        return module.emitError(
            "invalid async persistent conflict relationship");
  }
  for (const auto &[requestId, conflicts] : conflictsByRequest)
    for (const std::string &conflict : conflicts)
      if (!conflictsByRequest.at(conflict).count(requestId))
        return module.emitError(
            "asymmetric async persistent conflict relationship");
  SmallVector<Operation *> erase;
  for (const PersistentMaterialization &request : normalizedRequests) {
    for (udonhl::SpillLoadOp reload : request.reloads) {
      reload.getValue().replaceAllUsesWith(request.definition);
      erase.push_back(reload);
    }
    erase.push_back(request.store);
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
      [](Type type) -> FailureOr<std::string> { return hostStorageKey(type); },
      constants, true);
  if (failed(normalizedAnalysis)) {
    return failure();
  }
  std::map<std::string, std::string> valueIdByRequest;
  for (const PersistentMaterialization &request : normalizedRequests) {
    auto valueId = normalizedAnalysis->valueIds.find(request.definition);
    auto storageKey = normalizedAnalysis->storageKeys.find(request.definition);
    if (valueId == normalizedAnalysis->valueIds.end() ||
        storageKey == normalizedAnalysis->storageKeys.end() ||
        storageKey->second != request.physicalType.getValue()) {
      module.emitError("normalized async persistent value lacks identity");
      return failure();
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
        return failure();
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
  return success();
}

struct MaterializeAsyncPersistentStoragePass
    : impl::MaterializeAsyncPersistentStoragePassBase<
          MaterializeAsyncPersistentStoragePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    OwningOpRef<ModuleOp> candidate = cast<ModuleOp>(module->clone());
    if (failed(materializeAsyncPersistentStorage(*candidate))) {
      signalPassFailure();
      return;
    }
    module->setAttrs((*candidate)->getAttrDictionary());
    module.getBodyRegion().takeBody(candidate->getBodyRegion());
  }
};

} // namespace
} // namespace vrc
