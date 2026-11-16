#include "VRC/Conversion/Passes.h"
#include "VRC/Conversion/HostLiteralStorage.h"
#include "VRC/Analysis/HostStorageAllocation.h"
#include "VRC/Analysis/HostStorageLiveness.h"
#include "VRC/Support/HostLiteral.h"
#include "VRC/Support/HostStorageType.h"
#include "VRC/Support/UdonLimits.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "VRC/Dialect/VRC/VRCDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
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

struct ScratchAllocation {
  DictionaryAttr source;
  std::string instanceId;
  std::string consumerId;
  std::string literalKey;
  std::string logicalType;
  std::string physicalType;
  SmallVector<std::string> siteIds;
  std::optional<HostStorageNodeId> node;
};

struct AllocateHostStoragePass
    : impl::AllocateHostStoragePassBase<AllocateHostStoragePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    Builder builder(module.getContext());
    if (failed(allocateHostLiteralStorage(module))) {
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
    std::map<std::tuple<std::string, std::string, int32_t>, std::string>
        persistentRequestIds;
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
        persistentRequestIds.emplace(
            std::make_tuple(function.getSymName().str(), spill.storageKey,
                            spill.oldIndex),
            "function:" + function.getSymName().str() +
                ":async_persistent:" + std::to_string(spill.valueId));
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
      auto function = operation->getParentOfType<func::FuncOp>();
      auto request =
          function
              ? persistentRequestIds.find(std::make_tuple(
                    function.getSymName().str(), storageKey.getValue().str(),
                    static_cast<int32_t>(index.getInt())))
              : persistentRequestIds.end();
      if (request == persistentRequestIds.end()) {
        operation->emitError(
            "async frame access has no persistent request identity");
        invalid = true;
        return;
      }
      operation->setAttr("vrc.async_persistent_request_id",
                         builder.getStringAttr(request->second));
      operation->setAttr("index", builder.getI32IntegerAttr(mapped->second));
    });
    if (invalid)
      signalPassFailure();
  }

public:
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
    if (failed(allocateWritableCandidates(module, builder)))
      return failure();
    return selectPressurePlacements(module, builder);
  }

  LogicalResult selectPressurePlacements(ModuleOp module, Builder &builder) {
    auto request = module->getAttrOfType<DictionaryAttr>(
        "vrc.host_storage_pressure_request");
    bool hasSelection =
        module->hasAttr("vrc.host_literal_selected_placement_version") ||
        module->hasAttr("vrc.host_literal_selected_placements") ||
        module->hasAttr("vrc.host_storage_selection");
    if (!request)
      return hasSelection
                 ? module.emitError(
                       "Host storage selection has no pressure request")
                 : success();
    if (hasSelection)
      return module.emitError(
          "Host storage pressure selection already has an output");

    auto schema = request.getAs<IntegerAttr>("schema_version");
    auto measured = request.getAs<IntegerAttr>("measured_data_slots");
    auto capacity = request.getAs<IntegerAttr>("data_slot_capacity");
    auto required =
        request.getAs<IntegerAttr>("required_resident_slot_savings");
    auto isI32 = [](IntegerAttr value) {
      return value && value.getType().isInteger(32);
    };
    if (!isI32(schema) || schema.getInt() != 1 || !isI32(measured) ||
        !isI32(capacity) || capacity.getInt() < 1 ||
        capacity.getInt() > kUdonHeapMaximumCapacity ||
        measured.getInt() <= capacity.getInt() || !isI32(required) ||
        required.getInt() <= 0 ||
        required.getInt() != measured.getInt() - capacity.getInt())
      return module.emitError("invalid Host storage pressure request");

    auto uses = module->getAttrOfType<ArrayAttr>("vrc.host_literal_uses");
    auto sites = module->getAttrOfType<ArrayAttr>("vrc.host_literal_use_sites");
    auto scratchAllocations = module->getAttrOfType<ArrayAttr>(
        "vrc.host_literal_scratch_allocations");
    auto residentAllocations =
        module->getAttrOfType<ArrayAttr>("vrc.host_literal_allocations");
    if (!uses || !sites || !scratchAllocations || !residentAllocations)
      return module.emitError(
          "Host storage pressure selection lacks allocation inputs");

    struct LiteralUses {
      uint64_t count = 0;
      bool onlyStaticResourceIndices = true;
    };
    std::map<std::string, LiteralUses> usesByKey;
    for (Attribute attribute : uses) {
      auto use = dyn_cast<DictionaryAttr>(attribute);
      auto key = use ? use.getAs<StringAttr>("literal_key") : StringAttr();
      auto kind = use ? use.getAs<StringAttr>("kind") : StringAttr();
      if (!key || !kind)
        return module.emitError(
            "invalid Host literal use during pressure selection");
      LiteralUses &summary = usesByKey[key.getValue().str()];
      ++summary.count;
      summary.onlyStaticResourceIndices &=
          kind.getValue() == "static_resource_index";
    }

    std::map<std::string, std::string> roleBySite;
    for (Attribute attribute : sites) {
      auto site = dyn_cast<DictionaryAttr>(attribute);
      auto id = site ? site.getAs<StringAttr>("site_id") : StringAttr();
      auto role = site ? site.getAs<StringAttr>("value_role") : StringAttr();
      if (!id || !role ||
          !roleBySite.emplace(id.getValue().str(), role.getValue().str())
               .second)
        return module.emitError(
            "invalid Host literal site during pressure selection");
    }

    struct ResidentBinding {
      int64_t color;
    };
    std::map<std::string, ResidentBinding> residentByKey;
    for (Attribute attribute : residentAllocations) {
      auto allocation = dyn_cast<DictionaryAttr>(attribute);
      auto key =
          allocation ? allocation.getAs<StringAttr>("key_id") : StringAttr();
      auto type = allocation ? allocation.getAs<StringAttr>("physical_type")
                             : StringAttr();
      auto placement =
          allocation ? allocation.getAs<StringAttr>("placement") : StringAttr();
      auto color =
          allocation ? allocation.getAs<IntegerAttr>("color") : IntegerAttr();
      if (!key || !type || type.getValue() != "SystemInt32" || !placement ||
          placement.getValue() != "resident_shared" || !color ||
          color.getInt() < 0 ||
          !residentByKey
               .emplace(key.getValue().str(), ResidentBinding{color.getInt()})
               .second)
        continue;
    }

    struct Instance {
      DictionaryAttr allocation;
      std::string instanceId;
      std::string literalKey;
      ArrayAttr siteIds;
      int64_t color;
    };
    std::map<std::string, SmallVector<Instance>> instancesByKey;
    std::map<std::string, uint64_t> siteCountByKey;
    for (Attribute attribute : scratchAllocations) {
      auto allocation = dyn_cast<DictionaryAttr>(attribute);
      auto instance = allocation ? allocation.getAs<StringAttr>("instance_id")
                                 : StringAttr();
      auto key = allocation ? allocation.getAs<StringAttr>("literal_key")
                            : StringAttr();
      auto logical = allocation ? allocation.getAs<StringAttr>("logical_type")
                                : StringAttr();
      auto physical = allocation ? allocation.getAs<StringAttr>("physical_type")
                                 : StringAttr();
      auto siteIds =
          allocation ? allocation.getAs<ArrayAttr>("site_ids") : ArrayAttr();
      auto color =
          allocation ? allocation.getAs<IntegerAttr>("color") : IntegerAttr();
      bool allStatic = allocation && instance && key && logical &&
                       logical.getValue() == "i32" && physical &&
                       physical.getValue() == "SystemInt32" && siteIds &&
                       !siteIds.empty() && color && color.getInt() >= 0;
      if (allStatic)
        for (Attribute siteAttribute : siteIds) {
          auto site = dyn_cast<StringAttr>(siteAttribute);
          auto role =
              site ? roleBySite.find(site.getValue().str()) : roleBySite.end();
          allStatic &= site && role != roleBySite.end() &&
                       role->second == "static_resource_index";
        }
      if (!allStatic)
        continue;
      instancesByKey[key.getValue().str()].push_back(
          {allocation, instance.getValue().str(), key.getValue().str(), siteIds,
           color.getInt()});
      siteCountByKey[key.getValue().str()] += siteIds.size();
    }

    struct Candidate {
      std::string literalKey;
      SmallVector<Instance> instances;
      HostLiteralI32Recipe recipe;
      uint64_t codeBytes = 0;
      SmallVector<std::string> seedKeys;
    };
    SmallVector<Candidate, 0> candidates;
    std::set<std::string> allSeedKeys;
    for (auto &[key, instances] : instancesByKey) {
      auto usesFound = usesByKey.find(key);
      if (usesFound == usesByKey.end() ||
          !usesFound->second.onlyStaticResourceIndices ||
          usesFound->second.count != siteCountByKey[key] ||
          !StringRef(key).starts_with("SystemInt32:0x"))
        continue;
      uint64_t bits = 0;
      if (StringRef(key)
              .drop_front(sizeof("SystemInt32:0x") - 1)
              .getAsInteger(16, bits) ||
          bits > std::numeric_limits<uint32_t>::max())
        continue;
      auto recipe = getHostLiteralI32Recipe(
          builder.getI32Type(),
          builder.getI32IntegerAttr(static_cast<int32_t>(uint32_t(bits))));
      if (!recipe)
        continue;
      Candidate candidate{key, instances, *recipe};
      bool seedsAvailable = true;
      SmallVector<std::string> candidateSeedKeys;
      for (unsigned seed : recipe->seedValues) {
        HostLiteralKey seedKey = getHostLiteralKey(
            builder.getI32Type(), builder.getI32IntegerAttr(seed));
        if (!residentByKey.count(seedKey.id))
          seedsAvailable = false;
        candidateSeedKeys.push_back(seedKey.id);
      }
      if (!seedsAvailable)
        continue;
      candidate.seedKeys = std::move(candidateSeedKeys);
      allSeedKeys.insert(candidate.seedKeys.begin(), candidate.seedKeys.end());
      uint32_t value = static_cast<uint32_t>(bits);
      unsigned highestNibble = 0;
      for (unsigned nibble = 1; nibble < 8; ++nibble)
        if ((value >> (nibble * 4)) & 15)
          highestNibble = nibble;
      unsigned lowerNonzero = 0;
      for (unsigned nibble = 0; nibble < highestNibble; ++nibble)
        lowerNonzero += ((value >> (nibble * 4)) & 15) != 0;
      candidate.codeBytes =
          instances.size() * (20 + 32 * (highestNibble + lowerNonzero));
      llvm::sort(candidate.instances,
                 [](const Instance &left, const Instance &right) {
                   return left.instanceId < right.instanceId;
                 });
      candidates.push_back(std::move(candidate));
    }
    llvm::erase_if(candidates, [&](const Candidate &candidate) {
      return allSeedKeys.count(candidate.literalKey);
    });
    llvm::sort(candidates, [](const Candidate &left, const Candidate &right) {
      return std::tie(left.codeBytes, left.instances.front().instanceId,
                      left.literalKey) <
             std::tie(right.codeBytes, right.instances.front().instanceId,
                      right.literalKey);
    });
    if (static_cast<uint64_t>(required.getInt()) > candidates.size())
      return module.emitError()
             << "Host storage pressure requires " << required.getInt()
             << " resident slots but only " << candidates.size()
             << " are safely selectable";

    SmallVector<Attribute> selections;
    SmallVector<const Candidate *> selected;
    for (int64_t index = 0; index < required.getInt(); ++index)
      selected.push_back(&candidates[index]);
    SmallVector<const Instance *> selectedInstances;
    for (const Candidate *candidate : selected)
      for (const Instance &instance : candidate->instances)
        selectedInstances.push_back(&instance);
    llvm::sort(selectedInstances,
               [](const Instance *left, const Instance *right) {
                 return left->instanceId < right->instanceId;
               });
    for (const Instance *instance : selectedInstances) {
      auto candidateIt = llvm::find_if(selected, [&](const Candidate *item) {
        return item->literalKey == instance->literalKey;
      });
      const Candidate &candidate = **candidateIt;
      SmallVector<Attribute> seedBindings;
      for (const std::string &seedKey : candidate.seedKeys) {
        NamedAttrList seed;
        seed.set("literal_key", builder.getStringAttr(seedKey));
        seed.set("physical_type", builder.getStringAttr("SystemInt32"));
        seed.set("storage_domain", builder.getStringAttr("resident"));
        seed.set("color",
                 builder.getI32IntegerAttr(residentByKey.at(seedKey).color));
        seedBindings.push_back(builder.getDictionaryAttr(seed));
      }
      SmallVector<Attribute> externs;
      for (StringRef name : candidate.recipe.requiredExterns)
        externs.push_back(builder.getStringAttr(name));
      NamedAttrList fields;
      fields.set("instance_id", builder.getStringAttr(instance->instanceId));
      fields.set("site_ids", instance->siteIds);
      fields.set("literal_key", builder.getStringAttr(instance->literalKey));
      fields.set("logical_type", builder.getStringAttr("i32"));
      fields.set("physical_type", builder.getStringAttr("SystemInt32"));
      fields.set("placement", builder.getStringAttr("rematerialized"));
      fields.set("storage_domain", builder.getStringAttr("writable"));
      fields.set("color", builder.getI32IntegerAttr(instance->color));
      fields.set("recipe", builder.getStringAttr("rematerialize_i32"));
      fields.set("generated_version", builder.getI32IntegerAttr(1));
      fields.set("helper_scratch", builder.getI32IntegerAttr(0));
      fields.set("required_externs", builder.getArrayAttr(externs));
      fields.set("seed_bindings", builder.getArrayAttr(seedBindings));
      selections.push_back(builder.getDictionaryAttr(fields));
    }
    module->setAttr("vrc.host_literal_selected_placement_version",
                    builder.getI32IntegerAttr(1));
    module->setAttr("vrc.host_literal_selected_placements",
                    builder.getArrayAttr(selections));
    module->setAttr(
        "vrc.host_storage_selection",
        builder.getDictionaryAttr({
            builder.getNamedAttr("schema_version",
                                 builder.getI32IntegerAttr(1)),
            builder.getNamedAttr("required_resident_slot_savings",
                                 builder.getI32IntegerAttr(required.getInt())),
            builder.getNamedAttr("guaranteed_resident_slot_savings",
                                 builder.getI32IntegerAttr(selected.size())),
            builder.getNamedAttr("selected_literal_key_count",
                                 builder.getI32IntegerAttr(selected.size())),
        }));
    return success();
  }

  LogicalResult allocateWritableCandidates(ModuleOp module, Builder &builder) {
    auto version =
        module->getAttrOfType<IntegerAttr>("vrc.host_literal_use_site_version");
    auto sites = module->getAttrOfType<ArrayAttr>("vrc.host_literal_use_sites");
    auto instances =
        module->getAttrOfType<ArrayAttr>("vrc.host_literal_scratch_instances");
    auto groups =
        module->getAttrOfType<ArrayAttr>("vrc.host_literal_consumer_groups");
    if (!version || version.getInt() != 3 || !sites || !instances || !groups)
      return module.emitError("invalid Host literal use-site schema");

    SmallVector<func::FuncOp> functions;
    for (func::FuncOp function : module.getOps<func::FuncOp>())
      if (!function.isExternal())
        functions.push_back(function);
    llvm::sort(functions, [](func::FuncOp left, func::FuncOp right) {
      return left.getSymName() < right.getSymName();
    });
    llvm::DenseSet<Operation *> materializedConstants;
    auto analysis = analyzeHostStorageInterference(
        module, functions,
        [](Type type) -> FailureOr<std::string> {
          return hostStorageKey(type);
        },
        materializedConstants, true);
    if (failed(analysis))
      return failure();

    std::map<std::string, Operation *> operations;
    for (const auto &[operation, stableId] : analysis->operationIds)
      if (!operations.emplace(stableId, operation).second)
        return module.emitError("duplicate Host operation identity");

    HostStorageProblem problem;
    DenseMap<Value, HostStorageNodeId> directNodes;
    DenseMap<Value, HostStorageAlternativeId> directAlternatives;
    DenseMap<Value, uint64_t> directPriorities;
    SmallVector<Value> directValues;
    for (const auto &[value, ordinal] : analysis->ordinals)
      directValues.push_back(value);
    llvm::sort(directValues, [&](Value left, Value right) {
      return analysis->ordinals.lookup(left) < analysis->ordinals.lookup(right);
    });
    for (Value value : directValues) {
      auto stableId = analysis->valueIds.find(value);
      auto physicalType = analysis->storageKeys.find(value);
      if (stableId == analysis->valueIds.end() ||
          physicalType == analysis->storageKeys.end())
        return module.emitError("Host direct value lacks allocation identity");
      uint64_t priority = 1;
      for (OpOperand &use : value.getUses())
        if (priority != std::numeric_limits<uint64_t>::max())
          ++priority;
      auto node =
          problem.addRequest("direct:" + stableId->second, physicalType->second,
                             analysis->ordinals.lookup(value), priority);
      if (!node)
        return module.emitError("duplicate Host direct allocation identity");
      auto alternative = problem.addAlternative(
          *node, "direct", HostStoragePlacementKind::Direct,
          HostStorageDomain::Writable, true, 0, 0);
      if (!alternative)
        return module.emitError("invalid Host direct allocation alternative");
      directNodes.try_emplace(value, *node);
      directAlternatives.try_emplace(value, *alternative);
      directPriorities.try_emplace(value, priority);
    }
    for (Value value : directValues)
      for (Value conflict : analysis->interference.lookup(value)) {
        auto found = directNodes.find(conflict);
        if (found != directNodes.end() &&
            !problem.addInterference(directNodes.lookup(value), found->second))
          return module.emitError("invalid Host direct interference");
      }

    std::set<std::string> knownSiteIds;
    std::map<std::string, std::string> instanceBySite;
    std::map<std::string, std::string> groupByInstance;
    std::map<std::string, std::string> consumerBySite;
    std::map<std::string, std::string> literalBySite;
    std::map<std::string, std::string> logicalTypeBySite;
    std::map<std::string, std::string> physicalTypeBySite;
    for (Attribute attribute : sites) {
      auto site = dyn_cast<DictionaryAttr>(attribute);
      auto siteId = site ? site.getAs<StringAttr>("site_id") : StringAttr();
      auto eligible =
          site ? site.getAs<BoolAttr>("scratch_eligible") : BoolAttr();
      auto instance =
          site ? site.getAs<StringAttr>("scratch_instance_id") : StringAttr();
      auto group =
          site ? site.getAs<StringAttr>("consumer_group_id") : StringAttr();
      auto consumer =
          site ? site.getAs<StringAttr>("consumer_id") : StringAttr();
      auto literal =
          site ? site.getAs<StringAttr>("literal_key") : StringAttr();
      auto physicalType =
          site ? site.getAs<StringAttr>("physical_type") : StringAttr();
      auto definition =
          site ? site.getAs<StringAttr>("definition_use_id") : StringAttr();
      auto operandIndex =
          site ? site.getAs<IntegerAttr>("operand_index") : IntegerAttr();
      auto logicalType =
          site ? site.getAs<StringAttr>("logical_type") : StringAttr();
      auto role = site ? site.getAs<StringAttr>("value_role") : StringAttr();
      auto resourceIndex =
          site ? site.getAs<IntegerAttr>("resource_index") : IntegerAttr();
      auto lifetime = site ? site.getAs<StringAttr>("lifetime") : StringAttr();
      auto crossesCall =
          site ? site.getAs<BoolAttr>("crosses_call") : BoolAttr();
      auto crossesSuspend =
          site ? site.getAs<BoolAttr>("crosses_suspend") : BoolAttr();
      auto operation = consumer ? operations.find(consumer.getValue().str())
                                : operations.end();
      bool validOperand = operation != operations.end() && operandIndex &&
                          operandIndex.getInt() >= 0 &&
                          static_cast<uint64_t>(operandIndex.getInt()) <
                              operation->second->getNumOperands();
      arith::ConstantOp constant;
      HostLiteralKey key;
      if (validOperand) {
        Value operand = operation->second->getOperand(operandIndex.getInt());
        constant = operand.getDefiningOp<arith::ConstantOp>();
        if (constant)
          key = getHostLiteralKey(constant.getType(), constant.getValue());
      }
      auto generatedRole =
          constant
              ? constant->getAttrOfType<StringAttr>("vrc.host_literal_role")
              : StringAttr();
      auto dispatch = validOperand
                          ? dyn_cast<::vrc::vrc::DispatchOp>(operation->second)
                          : ::vrc::vrc::DispatchOp();
      unsigned staticSegmentStart =
          dispatch ? 1 + dispatch.getResources().size() : 0;
      bool validStaticResourceIndex = false;
      if (dispatch && operandIndex && resourceIndex &&
          operandIndex.getInt() >= staticSegmentStart) {
        unsigned segmentIndex =
            static_cast<unsigned>(operandIndex.getInt()) - staticSegmentStart;
        auto positions = dispatch.getStaticResourcePositions();
        validStaticResourceIndex =
            segmentIndex < dispatch.getStaticResourceIndices().size() &&
            positions && segmentIndex < positions->size() &&
            (*positions)[segmentIndex] == resourceIndex.getInt();
      }
      bool validRole =
          role &&
          ((role == "source_operand" && !generatedRole && !resourceIndex) ||
           (role == "static_resource_index" &&
            generatedRole == "static_resource_index" && resourceIndex &&
            resourceIndex.getInt() >= 0 && validStaticResourceIndex));
      bool expectedEligible =
          constant &&
          llvm::any_of(key.contextFreeRecipes,
                       [](StringRef recipe) {
                         return recipe.starts_with("rematerialize_");
                       }) &&
          !isa<cf::BranchOp, cf::CondBranchOp, cf::SwitchOp>(
              *operation->second);
      std::string expectedSiteId;
      if (consumer && operandIndex)
        expectedSiteId = consumer.getValue().str() +
                         ":operand:" + std::to_string(operandIndex.getInt());
      if (!site || !siteId || siteId.getValue().empty() || !eligible ||
          !consumer || !definition || !operandIndex || !logicalType ||
          !validRole || !validOperand || !constant ||
          siteId.getValue() != expectedSiteId ||
          definition.getValue() !=
              analysis->valueIds.lookup(constant.getResult()) ||
          !literal || literal.getValue().empty() || !physicalType ||
          literal.getValue() != key.id ||
          physicalType.getValue() != key.physicalType ||
          logicalType.getValue() != hostLiteralTypeText(constant.getType()) ||
          eligible.getValue() != expectedEligible || !lifetime ||
          lifetime.getValue() != "immediate_consumer" || !crossesCall ||
          crossesCall.getValue() || !crossesSuspend ||
          crossesSuspend.getValue() ||
          !knownSiteIds.insert(siteId.getValue().str()).second ||
          (eligible.getValue() && (!instance || !group)) ||
          (!eligible.getValue() && (instance || group)))
        return module.emitError("invalid Host literal use-site entry");
      if (eligible.getValue()) {
        instanceBySite.emplace(siteId.getValue().str(),
                               instance.getValue().str());
        consumerBySite.emplace(siteId.getValue().str(),
                               consumer.getValue().str());
        literalBySite.emplace(siteId.getValue().str(),
                              literal.getValue().str());
        logicalTypeBySite.emplace(siteId.getValue().str(),
                                  logicalType.getValue().str());
        physicalTypeBySite.emplace(siteId.getValue().str(),
                                   physicalType.getValue().str());
        auto [found, inserted] = groupByInstance.emplace(
            instance.getValue().str(), group.getValue().str());
        if (!inserted && found->second != group.getValue())
          return module.emitError(
              "Host scratch instance references multiple consumer groups");
      }
    }

    std::map<std::string, ScratchAllocation> scratch;
    std::set<std::string> coveredSites;
    uint64_t nextOrdinal = directValues.size();
    for (Attribute attribute : instances) {
      auto source = dyn_cast<DictionaryAttr>(attribute);
      auto instanceId =
          source ? source.getAs<StringAttr>("instance_id") : StringAttr();
      auto consumerId =
          source ? source.getAs<StringAttr>("consumer_id") : StringAttr();
      auto literalKey =
          source ? source.getAs<StringAttr>("literal_key") : StringAttr();
      auto logicalType =
          source ? source.getAs<StringAttr>("logical_type") : StringAttr();
      auto physicalType =
          source ? source.getAs<StringAttr>("physical_type") : StringAttr();
      auto sourceSites =
          source ? source.getAs<ArrayAttr>("site_ids") : ArrayAttr();
      if (!source || !instanceId || instanceId.getValue().empty() ||
          !consumerId ||
          operations.find(consumerId.getValue().str()) == operations.end() ||
          !literalKey || literalKey.getValue().empty() || !logicalType ||
          logicalType.getValue().empty() || !physicalType ||
          physicalType.getValue().empty() || !sourceSites ||
          scratch.count(instanceId.getValue().str()))
        return module.emitError("invalid Host literal scratch instance");
      std::string expectedInstanceId =
          consumerId.getValue().str() +
          ":literal:" + literalKey.getValue().str() +
          ":logical:" + logicalType.getValue().str();
      if (instanceId.getValue() != expectedInstanceId)
        return module.emitError("noncanonical Host scratch instance identity");
      ScratchAllocation allocation{source,
                                   instanceId.getValue().str(),
                                   consumerId.getValue().str(),
                                   literalKey.getValue().str(),
                                   logicalType.getValue().str(),
                                   physicalType.getValue().str(),
                                   {}};
      for (Attribute siteAttribute : sourceSites) {
        auto siteId = dyn_cast<StringAttr>(siteAttribute);
        if (!siteId ||
            instanceBySite.find(siteId.getValue().str()) ==
                instanceBySite.end() ||
            instanceBySite.at(siteId.getValue().str()) !=
                allocation.instanceId ||
            consumerBySite.at(siteId.getValue().str()) !=
                allocation.consumerId ||
            literalBySite.at(siteId.getValue().str()) !=
                allocation.literalKey ||
            logicalTypeBySite.at(siteId.getValue().str()) !=
                allocation.logicalType ||
            physicalTypeBySite.at(siteId.getValue().str()) !=
                allocation.physicalType ||
            !coveredSites.insert(siteId.getValue().str()).second ||
            llvm::is_contained(allocation.siteIds, siteId.getValue().str()))
          return module.emitError("invalid Host literal scratch site list");
        allocation.siteIds.push_back(siteId.getValue().str());
      }
      if (allocation.siteIds.empty())
        return module.emitError("Host literal scratch instance has no sites");
      allocation.node = problem.addNode("scratch:" + allocation.instanceId,
                                        allocation.physicalType, nextOrdinal++,
                                        HostStorageDomain::Writable,
                                        HostStorageConstraint::DirectOnly);
      if (!allocation.node)
        return module.emitError("duplicate Host scratch allocation identity");
      scratch.emplace(allocation.instanceId, std::move(allocation));
    }
    if (scratch.size() != groupByInstance.size() ||
        coveredSites.size() != instanceBySite.size())
      return module.emitError("Host literal scratch references are incomplete");

    std::set<std::string> seenGroups;
    std::set<std::string> allocatedInstances;
    for (Attribute attribute : groups) {
      auto group = dyn_cast<DictionaryAttr>(attribute);
      auto groupId = group ? group.getAs<StringAttr>("group_id") : StringAttr();
      auto consumerId =
          group ? group.getAs<StringAttr>("consumer_id") : StringAttr();
      auto physicalType =
          group ? group.getAs<StringAttr>("physical_type") : StringAttr();
      auto members =
          group ? group.getAs<ArrayAttr>("member_instance_ids") : ArrayAttr();
      auto operation = consumerId ? operations.find(consumerId.getValue().str())
                                  : operations.end();
      if (!group || !groupId || groupId.getValue().empty() || !consumerId ||
          !physicalType || physicalType.getValue().empty() || !members ||
          operation == operations.end() ||
          !seenGroups.insert(groupId.getValue().str()).second)
        return module.emitError("invalid Host literal consumer group");
      std::string expectedGroupId = consumerId.getValue().str() +
                                    ":scratch:" + physicalType.getValue().str();
      if (groupId.getValue() != expectedGroupId)
        return module.emitError("noncanonical Host consumer group identity");
      SmallVector<HostStorageNodeId> scratchMembers;
      for (Attribute memberAttribute : members) {
        auto memberId = dyn_cast<StringAttr>(memberAttribute);
        auto member =
            memberId ? scratch.find(memberId.getValue().str()) : scratch.end();
        if (!memberId || member == scratch.end() ||
            member->second.consumerId != consumerId.getValue() ||
            member->second.physicalType != physicalType.getValue() ||
            groupByInstance.find(member->first) == groupByInstance.end() ||
            groupByInstance.at(member->first) != groupId.getValue() ||
            !allocatedInstances.insert(member->first).second)
          return module.emitError("invalid Host literal consumer group member");
        scratchMembers.push_back(*member->second.node);
      }
      if (scratchMembers.empty())
        return module.emitError("Host literal consumer group is empty");
      auto directGroups =
          analysis->simultaneousValueGroups.find(operation->second);
      if (directGroups == analysis->simultaneousValueGroups.end() ||
          directGroups->second.empty())
        return module.emitError("Host literal consumer lacks a liveness phase");
      for (const SmallVector<Value> &directGroup : directGroups->second) {
        SmallVector<HostStorageNodeId> simultaneous(scratchMembers.begin(),
                                                    scratchMembers.end());
        for (Value value : directGroup)
          if (analysis->storageKeys.lookup(value) == physicalType.getValue())
            simultaneous.push_back(directNodes.lookup(value));
        llvm::sort(simultaneous);
        simultaneous.erase(
            std::unique(simultaneous.begin(), simultaneous.end()),
            simultaneous.end());
        if (!problem.addSimultaneousGroup(simultaneous))
          return module.emitError("invalid Host literal simultaneous group");
      }
    }
    if (allocatedInstances.size() != scratch.size())
      return module.emitError("Host scratch instance has no consumer group");

    HostStorageSolution solution = solveHostStorageProblem(problem);
    if (!solution.success)
      return module.emitError("Host writable allocation has no legal solution");
    SmallVector<Attribute> directAllocations;
    for (Value value : directValues) {
      HostStorageNodeId node = directNodes.lookup(value);
      NamedAttrList fields;
      fields.set("value_id",
                 builder.getStringAttr(analysis->valueIds.lookup(value)));
      fields.set("physical_type",
                 builder.getStringAttr(analysis->storageKeys.lookup(value)));
      fields.set("storage_domain", builder.getStringAttr("writable"));
      fields.set("allocation", builder.getStringAttr("direct_color_candidate"));
      fields.set("priority",
                 builder.getI64IntegerAttr(directPriorities.lookup(value)));
      fields.set("selected_alternative", builder.getStringAttr("direct"));
      if (solution.getSelectedAlternative(node) !=
          directAlternatives.lookup(value))
        return module.emitError(
            "Host direct allocation selected a wrong alternative");
      fields.set("color", builder.getI32IntegerAttr(solution.getColor(node)));
      directAllocations.push_back(builder.getDictionaryAttr(fields));
    }
    SmallVector<Attribute> scratchAllocations;
    for (const auto &[instanceId, allocation] : scratch) {
      NamedAttrList fields(allocation.source);
      fields.set("storage_domain", builder.getStringAttr("writable"));
      fields.set("allocation",
                 builder.getStringAttr("immediate_consumer_candidate"));
      fields.set("color", builder.getI32IntegerAttr(
                              solution.getColor(*allocation.node)));
      scratchAllocations.push_back(builder.getDictionaryAttr(fields));
    }
    SmallVector<Attribute> extents;
    for (const auto &[key, extent] : solution.extents) {
      if (key.domain != HostStorageDomain::Writable)
        continue;
      NamedAttrList fields;
      fields.set("storage_domain", builder.getStringAttr("writable"));
      fields.set("physical_type", builder.getStringAttr(key.physicalType));
      fields.set("extent", builder.getI32IntegerAttr(extent));
      extents.push_back(builder.getDictionaryAttr(fields));
    }
    module->setAttr("vrc.host_writable_allocation_version",
                    builder.getI32IntegerAttr(2));
    module->setAttr("vrc.host_direct_color_allocations",
                    builder.getArrayAttr(directAllocations));
    module->setAttr("vrc.host_literal_scratch_allocations",
                    builder.getArrayAttr(scratchAllocations));
    module->setAttr("vrc.host_writable_extents", builder.getArrayAttr(extents));
    return success();
  }
};

} // namespace

LogicalResult allocateHostLiteralStorage(ModuleOp module) {
  Builder builder(module.getContext());
  AllocateHostStoragePass allocator;
  return allocator.allocateLiterals(module, builder);
}

} // namespace vrc
