#include "VRC/Conversion/Passes.h"
#include "VRC/Analysis/HostStorageAllocation.h"
#include "VRC/Analysis/HostStorageLiveness.h"
#include "VRC/Support/HostLiteral.h"
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
    return allocateWritableCandidates(module, builder);
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
          StringRef physicalType = hostLiteralPhysicalType(type);
          if (physicalType == "unsupported")
            return failure();
          return physicalType.str();
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
      auto node = problem.addNode(
          "direct:" + stableId->second, physicalType->second,
          analysis->ordinals.lookup(value), HostStorageDomain::Writable,
          HostStorageConstraint::DirectOnly);
      if (!node)
        return module.emitError("duplicate Host direct allocation identity");
      directNodes.try_emplace(value, *node);
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
} // namespace vrc
