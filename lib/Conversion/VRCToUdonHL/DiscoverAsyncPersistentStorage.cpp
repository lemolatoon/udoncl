#include "VRC/Conversion/Passes.h"
#include "VRC/Analysis/HostStorageLiveness.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "VRC/Support/HostStorageType.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include <limits>
#include <map>
#include <set>
#include <string>

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_DISCOVERASYNCPERSISTENTSTORAGEPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {

struct PersistentEntry {
  int32_t valueId;
  int32_t compatibilityIndex;
  std::string physicalType;
  SmallVector<int32_t> conflicts;
  Value definition;
  SmallVector<Value> reloads;
};

struct DiscoverAsyncPersistentStoragePass
    : impl::DiscoverAsyncPersistentStoragePassBase<
          DiscoverAsyncPersistentStoragePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<func::FuncOp> functions;
    for (func::FuncOp function : module.getOps<func::FuncOp>())
      if (!function.isExternal())
        functions.push_back(function);
    llvm::sort(functions, [](func::FuncOp left, func::FuncOp right) {
      return left.getSymName() < right.getSymName();
    });

    bool hasPlans = llvm::any_of(functions, [](func::FuncOp function) {
      return function->hasAttr("vrc.async_spill_slots");
    });
    if (!hasPlans) {
      module->removeAttr("vrc.async_persistent_request_version");
      module->removeAttr("vrc.async_persistent_requests");
      return;
    }

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

    Builder builder(module.getContext());
    SmallVector<Attribute> requests;
    bool invalid = false;
    for (func::FuncOp function : functions) {
      auto plan = function->getAttrOfType<ArrayAttr>("vrc.async_spill_slots");
      if (!plan)
        continue;

      std::map<std::string, PersistentEntry> entries;
      std::map<std::pair<std::string, int32_t>, SmallVector<std::string>>
          requestsByCompatibilityStorage;
      std::map<int32_t, std::string> identityByValue;
      std::string functionPrefix =
          "function:" + function.getSymName().str() + ":async_persistent:";
      for (Attribute attribute : plan) {
        auto entry = dyn_cast<DictionaryAttr>(attribute);
        auto valueId =
            entry ? entry.getAs<IntegerAttr>("value_id") : IntegerAttr();
        auto index = entry ? entry.getAs<IntegerAttr>("slot") : IntegerAttr();
        auto physicalType =
            entry ? entry.getAs<StringAttr>("storage_key") : StringAttr();
        auto conflicts = entry
                             ? entry.getAs<DenseI32ArrayAttr>("interferes_with")
                             : DenseI32ArrayAttr();
        if (!entry || !valueId || !index || !physicalType || !conflicts ||
            valueId.getInt() < 0 ||
            valueId.getInt() > std::numeric_limits<int32_t>::max() ||
            index.getInt() < 0 ||
            index.getInt() > std::numeric_limits<int32_t>::max() ||
            physicalType.getValue().empty()) {
          function.emitError(
              "invalid async persistent storage discovery input");
          invalid = true;
          break;
        }
        int32_t id = static_cast<int32_t>(valueId.getInt());
        int32_t compatibilityIndex = static_cast<int32_t>(index.getInt());
        auto key =
            std::make_pair(physicalType.getValue().str(), compatibilityIndex);
        std::string requestId = functionPrefix + std::to_string(id);
        PersistentEntry persistent{id,
                                   compatibilityIndex,
                                   physicalType.getValue().str(),
                                   SmallVector<int32_t>(conflicts.asArrayRef()),
                                   {},
                                   {}};
        if (!entries.emplace(requestId, std::move(persistent)).second ||
            !identityByValue.emplace(id, requestId).second) {
          function.emitError("duplicate async persistent storage identity");
          invalid = true;
          break;
        }
        requestsByCompatibilityStorage[key].push_back(std::move(requestId));
      }
      if (invalid)
        break;

      function.walk([&](Operation *operation) {
        if (invalid || (!operation->hasAttr("vrc.async_frame_store") &&
                        !operation->hasAttr("vrc.async_frame_load")))
          return;
        auto physicalType = operation->getAttrOfType<StringAttr>("storage_key");
        auto index = operation->getAttrOfType<IntegerAttr>("index");
        if (!physicalType || !index || index.getInt() < 0 ||
            index.getInt() > std::numeric_limits<int32_t>::max()) {
          operation->emitError("invalid async persistent frame access");
          invalid = true;
          return;
        }
        auto requestId = operation->getAttrOfType<StringAttr>(
            "vrc.async_persistent_request_id");
        std::string request;
        if (requestId) {
          request = requestId.getValue().str();
        } else {
          auto candidates = requestsByCompatibilityStorage.find(
              {physicalType.getValue().str(),
               static_cast<int32_t>(index.getInt())});
          if (candidates != requestsByCompatibilityStorage.end() &&
              candidates->second.size() == 1)
            request = candidates->second.front();
        }
        auto found = entries.find(request);
        if (found == entries.end()) {
          operation->emitError("async persistent frame access has no request");
          invalid = true;
          return;
        }
        if (auto store = dyn_cast<udonhl::SpillStoreOp>(operation)) {
          if (found->second.definition) {
            store.emitError(
                "async persistent request has multiple definitions");
            invalid = true;
            return;
          }
          found->second.definition = store.getValue();
        } else if (auto load = dyn_cast<udonhl::SpillLoadOp>(operation)) {
          found->second.reloads.push_back(load.getValue());
        } else {
          operation->emitError("async frame marker is not a spill access");
          invalid = true;
        }
      });
      if (invalid)
        break;

      for (auto &[requestId, entry] : entries) {
        if (!entry.definition || entry.reloads.empty()) {
          function.emitError(
              "async persistent request lacks definition or reload");
          invalid = true;
          break;
        }
        auto definitionId = analysis->valueIds.find(entry.definition);
        auto definitionType = hostStorageKey(entry.definition.getType());
        if (definitionId == analysis->valueIds.end() ||
            failed(definitionType) || *definitionType != entry.physicalType) {
          function.emitError(
              "async persistent definition lacks stable identity");
          invalid = true;
          break;
        }
        SmallVector<std::string> memberIds{definitionId->second};
        uint64_t priority = 1;
        for (Value reload : entry.reloads) {
          auto reloadId = analysis->valueIds.find(reload);
          auto reloadType = hostStorageKey(reload.getType());
          if (reloadId == analysis->valueIds.end() || failed(reloadType) ||
              *reloadType != entry.physicalType) {
            function.emitError("async persistent reload lacks stable identity");
            invalid = true;
            break;
          }
          memberIds.push_back(reloadId->second);
          for (OpOperand &use : reload.getUses())
            if (!use.getOwner()->hasAttr("vrc.async_frame_store") &&
                !use.getOwner()->hasAttr("vrc.async_frame_load") &&
                priority != std::numeric_limits<uint64_t>::max())
              ++priority;
        }
        if (invalid)
          break;
        llvm::sort(memberIds);
        memberIds.erase(std::unique(memberIds.begin(), memberIds.end()),
                        memberIds.end());
        SmallVector<Attribute> members;
        for (const std::string &member : memberIds)
          members.push_back(builder.getStringAttr(member));

        SmallVector<Attribute> conflicts;
        for (int32_t conflict : entry.conflicts) {
          auto found = identityByValue.find(conflict);
          if (found == identityByValue.end()) {
            function.emitError(
                "async persistent request references an unknown conflict");
            invalid = true;
            break;
          }
          conflicts.push_back(builder.getStringAttr(found->second));
        }
        if (invalid)
          break;
        llvm::sort(conflicts, [](Attribute left, Attribute right) {
          return cast<StringAttr>(left).getValue() <
                 cast<StringAttr>(right).getValue();
        });

        requests.push_back(builder.getDictionaryAttr({
            builder.getNamedAttr("request_id",
                                 builder.getStringAttr(requestId)),
            builder.getNamedAttr("physical_type",
                                 builder.getStringAttr(entry.physicalType)),
            builder.getNamedAttr("priority",
                                 builder.getI64IntegerAttr(priority)),
            builder.getNamedAttr(
                "legal_alternatives",
                builder.getArrayAttr({builder.getStringAttr("direct")})),
            builder.getNamedAttr("member_value_ids",
                                 builder.getArrayAttr(members)),
            builder.getNamedAttr("interferes_with",
                                 builder.getArrayAttr(conflicts)),
            builder.getNamedAttr(
                "compatibility_index",
                builder.getI32IntegerAttr(entry.compatibilityIndex)),
        }));
      }
      if (invalid)
        break;
    }
    if (invalid) {
      signalPassFailure();
      return;
    }

    llvm::sort(requests, [](Attribute left, Attribute right) {
      return cast<DictionaryAttr>(left)
                 .getAs<StringAttr>("request_id")
                 .getValue() < cast<DictionaryAttr>(right)
                                   .getAs<StringAttr>("request_id")
                                   .getValue();
    });
    module->setAttr("vrc.async_persistent_request_version",
                    builder.getI32IntegerAttr(1));
    module->setAttr("vrc.async_persistent_requests",
                    builder.getArrayAttr(requests));
  }
};

} // namespace
} // namespace vrc
