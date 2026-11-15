#include "VRC/Conversion/Passes.h"
#include "VRC/Analysis/HostStorageAllocation.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include <limits>
#include <map>
#include <set>
#include <string>

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_ALLOCATEHOSTSTORAGEPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {

struct PlannedSpill {
  DictionaryAttr source;
  int32_t valueId;
  int32_t oldIndex;
  std::string storageKey;
  SmallVector<int32_t> conflicts;
  int32_t newIndex = -1;
};

struct SpillIdentity {
  std::string storageKey;
  int32_t index;

  bool operator<(const SpillIdentity &other) const {
    return std::tie(storageKey, index) <
           std::tie(other.storageKey, other.index);
  }
};

struct LiteralAllocation {
  DictionaryAttr source;
  std::string keyId;
  std::string physicalType;
  bool supported;
  uint64_t useCount = 0;
  std::optional<HostStorageNodeId> node;
};

struct AllocateHostStoragePass
    : impl::AllocateHostStoragePassBase<AllocateHostStoragePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    Builder builder(module.getContext());
    if (failed(allocateLiterals(module, builder))) {
      signalPassFailure();
      return;
    }
    SmallVector<func::FuncOp> functions;
    for (func::FuncOp function : module.getOps<func::FuncOp>())
      if (function->hasAttr("vrc.async_spill_slots"))
        functions.push_back(function);
    llvm::sort(functions, [](func::FuncOp left, func::FuncOp right) {
      return left.getSymName() < right.getSymName();
    });

    std::map<std::string, int32_t> nextIndex;
    std::map<SpillIdentity, int32_t> remappedIndices;
    bool invalid = false;
    for (func::FuncOp function : functions) {
      auto plan = function->getAttrOfType<ArrayAttr>("vrc.async_spill_slots");
      SmallVector<PlannedSpill> spills;
      for (Attribute attribute : plan) {
        auto entry = dyn_cast<DictionaryAttr>(attribute);
        auto valueId = entry ? entry.getAs<IntegerAttr>("value_id") : nullptr;
        auto oldIndex = entry ? entry.getAs<IntegerAttr>("slot") : nullptr;
        auto storageKey =
            entry ? entry.getAs<StringAttr>("storage_key") : nullptr;
        auto conflicts =
            entry ? entry.getAs<DenseI32ArrayAttr>("interferes_with") : nullptr;
        if (!entry || !valueId || !oldIndex || !storageKey || !conflicts ||
            valueId.getInt() < 0 ||
            valueId.getInt() > std::numeric_limits<int32_t>::max() ||
            oldIndex.getInt() < 0 ||
            oldIndex.getInt() > std::numeric_limits<int32_t>::max() ||
            storageKey.getValue().empty() ||
            llvm::any_of(conflicts.asArrayRef(),
                         [](int32_t conflict) { return conflict < 0; })) {
          function.emitError(
              "async persistent storage plan lacks allocation inputs");
          invalid = true;
          break;
        }
        spills.push_back({entry, static_cast<int32_t>(valueId.getInt()),
                          static_cast<int32_t>(oldIndex.getInt()),
                          storageKey.getValue().str(),
                          SmallVector<int32_t>(conflicts.asArrayRef())});
      }
      if (invalid)
        break;
      llvm::sort(spills,
                 [](const PlannedSpill &left, const PlannedSpill &right) {
                   return std::tie(left.storageKey, left.valueId) <
                          std::tie(right.storageKey, right.valueId);
                 });

      HostStorageProblem problem;
      DenseMap<int32_t, HostStorageNodeId> nodeByValue;
      for (const PlannedSpill &spill : spills) {
        auto node =
            problem.addNode(std::to_string(spill.valueId), spill.storageKey,
                            spill.valueId, HostStorageDomain::TypedSpill,
                            HostStorageConstraint::TypedSpillOnly);
        if (!node) {
          function.emitError("duplicate async persistent storage identity");
          invalid = true;
          break;
        }
        nodeByValue[spill.valueId] = *node;
      }
      if (invalid)
        break;
      for (const PlannedSpill &spill : spills)
        for (int32_t conflict : spill.conflicts) {
          auto found = nodeByValue.find(conflict);
          if (found == nodeByValue.end() ||
              !problem.addInterference(nodeByValue.lookup(spill.valueId),
                                       found->second)) {
            function.emitError(
                "async persistent storage references an unknown conflict");
            invalid = true;
            break;
          }
        }
      if (invalid)
        break;
      HostStorageSolution solution = solveHostStorageProblem(problem);
      for (PlannedSpill &spill : spills) {
        unsigned color = solution.getColor(nodeByValue.lookup(spill.valueId));
        spill.newIndex = nextIndex[spill.storageKey] + color;
        remappedIndices[{spill.storageKey, spill.oldIndex}] = spill.newIndex;
      }
      for (const auto &[colorKey, extent] : solution.extents) {
        const std::string &storageKey = colorKey.physicalType;
        if (extent >
                static_cast<unsigned>(std::numeric_limits<int32_t>::max()) ||
            nextIndex[storageKey] > std::numeric_limits<int32_t>::max() -
                                        static_cast<int32_t>(extent)) {
          function.emitError("typed spill allocation exceeds i32 capacity");
          invalid = true;
          break;
        }
        nextIndex[storageKey] += extent;
      }
      if (invalid)
        break;

      SmallVector<Attribute> allocated;
      llvm::sort(spills,
                 [](const PlannedSpill &left, const PlannedSpill &right) {
                   return left.valueId < right.valueId;
                 });
      for (const PlannedSpill &spill : spills) {
        NamedAttrList fields(spill.source);
        fields.set("slot", builder.getI32IntegerAttr(spill.newIndex));
        fields.set("allocation", builder.getStringAttr("liveness_colored"));
        allocated.push_back(builder.getDictionaryAttr(fields));
      }
      function->setAttr("vrc.async_spill_slots",
                        builder.getArrayAttr(allocated));

      function.walk([&](Operation *operation) {
        auto live = operation->getAttrOfType<DenseI32ArrayAttr>(
            "vrc.async_live_spill_slots");
        if (!live)
          return;
        SmallVector<int32_t> mapped;
        for (int32_t oldIndex : live.asArrayRef()) {
          for (const PlannedSpill &spill : spills)
            if (spill.oldIndex == oldIndex)
              mapped.push_back(spill.newIndex);
        }
        llvm::sort(mapped);
        mapped.erase(std::unique(mapped.begin(), mapped.end()), mapped.end());
        operation->setAttr("vrc.async_live_spill_slots",
                           builder.getDenseI32ArrayAttr(mapped));
      });
    }
    if (invalid) {
      signalPassFailure();
      return;
    }

    module.walk([&](Operation *operation) {
      if (!operation->hasAttr("vrc.async_frame_store") &&
          !operation->hasAttr("vrc.async_frame_load"))
        return;
      auto storageKey = operation->getAttrOfType<StringAttr>("storage_key");
      auto index = operation->getAttrOfType<IntegerAttr>("index");
      if (!storageKey || !index) {
        operation->emitError("async frame access lacks typed spill identity");
        invalid = true;
        return;
      }
      auto mapped = remappedIndices.find(
          {storageKey.getValue().str(), static_cast<int32_t>(index.getInt())});
      if (mapped == remappedIndices.end()) {
        operation->emitError("async frame access has no allocation decision");
        invalid = true;
        return;
      }
      operation->setAttr("index", builder.getI32IntegerAttr(mapped->second));
    });
    if (invalid)
      signalPassFailure();
  }

private:
  LogicalResult allocateLiterals(ModuleOp module, Builder &builder) {
    auto keys = module->getAttrOfType<ArrayAttr>("vrc.host_literal_keys");
    auto uses = module->getAttrOfType<ArrayAttr>("vrc.host_literal_uses");
    auto version = module->getAttrOfType<IntegerAttr>(
        "vrc.host_literal_inventory_version");
    if (!keys && !uses && !version) {
      if (module->hasAttr("vrc.host_literal_allocation_version") ||
          module->hasAttr("vrc.host_literal_allocations"))
        return module.emitError("Host literal allocation lacks its inventory");
      return success();
    }
    if (!keys || !uses || !version || version.getInt() != 1)
      return module.emitError("invalid Host literal inventory schema");

    std::map<std::string, LiteralAllocation> allocations;
    constexpr StringLiteral supportedPhysicalTypes[] = {
        "SystemBoolean", "SystemInt32", "SystemInt64", "SystemSingle"};
    for (Attribute attribute : keys) {
      auto source = dyn_cast<DictionaryAttr>(attribute);
      auto keyId = source ? source.getAs<StringAttr>("key_id") : StringAttr();
      auto physicalType =
          source ? source.getAs<StringAttr>("physical_type") : StringAttr();
      auto supported =
          source ? source.getAs<BoolAttr>("supported") : BoolAttr();
      auto value = source ? source.getAs<StringAttr>("value") : StringAttr();
      auto recipes = source ? source.getAs<ArrayAttr>("context_free_recipes")
                            : ArrayAttr();
      bool validRecipes = recipes != nullptr;
      bool hasResidentRecipe = false;
      std::set<std::string> recipeNames;
      if (recipes)
        for (Attribute recipe : recipes) {
          auto name = dyn_cast<StringAttr>(recipe);
          validRecipes &= name && !name.getValue().empty() &&
                          recipeNames.insert(name.getValue().str()).second;
          hasResidentRecipe |= name && name.getValue() == "resident_shared";
        }
      bool supportedContract = false;
      if (keyId && physicalType && supported && value && validRecipes &&
          !value.getValue().empty()) {
        std::string supportedKey =
            (physicalType.getValue() + ":" + value.getValue()).str();
        std::string unsupportedSuffix = (":" + value.getValue()).str();
        supportedContract =
            (supported.getValue() &&
             llvm::is_contained(supportedPhysicalTypes,
                                physicalType.getValue()) &&
             keyId.getValue() == supportedKey && hasResidentRecipe) ||
            (!supported.getValue() &&
             physicalType.getValue() == "unsupported" &&
             keyId.getValue().starts_with("unsupported:") &&
             keyId.getValue().ends_with(unsupportedSuffix) &&
             recipeNames.empty());
      }
      if (!source || !keyId || keyId.getValue().empty() || !physicalType ||
          physicalType.getValue().empty() || !value ||
          value.getValue().empty() || !validRecipes || !supportedContract ||
          allocations.count(keyId.getValue().str()))
        return module.emitError("invalid Host literal key inventory");
      allocations.emplace(keyId.getValue().str(),
                          LiteralAllocation{source, keyId.getValue().str(),
                                            physicalType.getValue().str(),
                                            supported.getValue()});
    }
    std::set<std::string> useIds;
    for (Attribute attribute : uses) {
      auto use = dyn_cast<DictionaryAttr>(attribute);
      auto key = use ? use.getAs<StringAttr>("literal_key") : StringAttr();
      auto useId = use ? use.getAs<StringAttr>("use_id") : StringAttr();
      auto origin = use ? use.getAs<StringAttr>("origin") : StringAttr();
      auto kind = use ? use.getAs<StringAttr>("kind") : StringAttr();
      auto logicalType =
          use ? use.getAs<StringAttr>("logical_type") : StringAttr();
      auto found =
          key ? allocations.find(key.getValue().str()) : allocations.end();
      if (!use || !key || !useId || useId.getValue().empty() || !origin ||
          origin.getValue().empty() || !kind || kind.getValue().empty() ||
          !logicalType || logicalType.getValue().empty() ||
          !useIds.insert(useId.getValue().str()).second ||
          found == allocations.end())
        return module.emitError("invalid Host literal use inventory");
      ++found->second.useCount;
    }

    HostStorageProblem problem;
    std::map<std::string, SmallVector<HostStorageNodeId>> nodesByType;
    uint64_t ordinal = 0;
    for (auto &[keyId, allocation] : allocations) {
      if (!allocation.useCount)
        return module.emitError(
            "Host literal inventory contains an orphan key");
      if (!allocation.supported)
        continue;
      allocation.node = problem.addNode(keyId, allocation.physicalType,
                                        ordinal++, HostStorageDomain::Resident,
                                        HostStorageConstraint::Fixed);
      if (!allocation.node)
        return module.emitError("invalid Host literal allocation identity");
      nodesByType[allocation.physicalType].push_back(*allocation.node);
    }
    for (const auto &[physicalType, nodes] : nodesByType)
      if (!problem.addSimultaneousGroup(nodes))
        return module.emitError("invalid Host literal resident group");
    HostStorageSolution solution = solveHostStorageProblem(problem);

    SmallVector<Attribute> result;
    for (const auto &[keyId, allocation] : allocations) {
      if (!allocation.supported)
        continue;
      NamedAttrList fields(allocation.source);
      fields.set("inventory_use_count",
                 builder.getI64IntegerAttr(allocation.useCount));
      fields.set("placement", builder.getStringAttr("resident_shared"));
      fields.set("reason",
                 builder.getStringAttr("compatibility_resident_baseline"));
      fields.set("storage_key", builder.getStringAttr(allocation.physicalType));
      fields.set("simultaneous_group",
                 builder.getStringAttr("resident:" + allocation.physicalType));
      if (allocation.node)
        fields.set("color", builder.getI32IntegerAttr(
                                solution.getColor(*allocation.node)));
      result.push_back(builder.getDictionaryAttr(fields));
    }
    module->setAttr("vrc.host_literal_allocation_version",
                    builder.getI32IntegerAttr(1));
    module->setAttr("vrc.host_literal_allocations",
                    builder.getArrayAttr(result));
    return success();
  }
};

} // namespace
} // namespace vrc
