#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "VRC/Support/HostLiteral.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include <functional>
#include <map>
#include <set>
#include <string>

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_MATERIALIZEHOSTSTORAGEPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {

struct MaterializeHostStoragePass
    : impl::MaterializeHostStoragePassBase<MaterializeHostStoragePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto selectionVersion = module->getAttrOfType<IntegerAttr>(
        "vrc.host_literal_selected_placement_version");
    auto selections = module->getAttrOfType<ArrayAttr>(
        "vrc.host_literal_selected_placements");
    if (!selectionVersion && !selections)
      return;
    if (!selectionVersion || selectionVersion.getInt() != 1 || !selections ||
        module->hasAttr("vrc.host_literal_materialization_version")) {
      module.emitError("invalid Host literal materialization selection schema");
      signalPassFailure();
      return;
    }

    auto useSiteVersion =
        module->getAttrOfType<IntegerAttr>("vrc.host_literal_use_site_version");
    auto writableVersion = module->getAttrOfType<IntegerAttr>(
        "vrc.host_writable_allocation_version");
    auto inventoryVersion = module->getAttrOfType<IntegerAttr>(
        "vrc.host_literal_inventory_version");
    auto allocationVersion = module->getAttrOfType<IntegerAttr>(
        "vrc.host_literal_allocation_version");
    auto sites = module->getAttrOfType<ArrayAttr>("vrc.host_literal_use_sites");
    auto instances =
        module->getAttrOfType<ArrayAttr>("vrc.host_literal_scratch_instances");
    auto scratchAllocations = module->getAttrOfType<ArrayAttr>(
        "vrc.host_literal_scratch_allocations");
    auto directAllocations =
        module->getAttrOfType<ArrayAttr>("vrc.host_direct_color_allocations");
    auto residentAllocations =
        module->getAttrOfType<ArrayAttr>("vrc.host_literal_allocations");
    auto writableExtents =
        module->getAttrOfType<ArrayAttr>("vrc.host_writable_extents");
    if (!useSiteVersion || useSiteVersion.getInt() != 2 || !writableVersion ||
        writableVersion.getInt() != 2 || !inventoryVersion ||
        inventoryVersion.getInt() != 1 || !allocationVersion ||
        allocationVersion.getInt() != 1 || !sites || !instances ||
        !scratchAllocations || !directAllocations || !residentAllocations ||
        !writableExtents) {
      module.emitError("Host literal selection lacks allocation inputs");
      signalPassFailure();
      return;
    }

    std::map<std::string, Operation *> operations;
    std::map<std::string, Value> values;
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      if (function.isExternal())
        continue;
      std::function<void(Region &, const std::string &)> collect;
      collect = [&](Region &region, const std::string &regionPath) {
        for (auto [blockIndex, block] : llvm::enumerate(region.getBlocks())) {
          std::string blockPath =
              regionPath + ":block:" + std::to_string(blockIndex);
          for (auto [argumentIndex, argument] :
               llvm::enumerate(block.getArguments()))
            values.emplace(blockPath +
                               ":argument:" + std::to_string(argumentIndex),
                           argument);
          for (auto [operationIndex, operation] : llvm::enumerate(block)) {
            std::string id =
                blockPath + ":operation:" + std::to_string(operationIndex);
            operations.emplace(id, &operation);
            for (auto [resultIndex, result] :
                 llvm::enumerate(operation.getResults()))
              values.emplace(id + ":result:" + std::to_string(resultIndex),
                             result);
            for (auto [regionIndex, nested] :
                 llvm::enumerate(operation.getRegions()))
              collect(nested, id + ":region:" + std::to_string(regionIndex));
          }
        }
      };
      collect(function.getBody(),
              "function:" + function.getSymName().str() + ":region:0");
    }

    SmallVector<std::pair<DictionaryAttr, Value>> directPlans;
    std::set<std::string> directIds;
    for (Attribute attribute : directAllocations) {
      auto allocation = dyn_cast<DictionaryAttr>(attribute);
      auto valueId =
          allocation ? allocation.getAs<StringAttr>("value_id") : StringAttr();
      auto physicalType = allocation
                              ? allocation.getAs<StringAttr>("physical_type")
                              : StringAttr();
      auto domain = allocation ? allocation.getAs<StringAttr>("storage_domain")
                               : StringAttr();
      auto kind = allocation ? allocation.getAs<StringAttr>("allocation")
                             : StringAttr();
      auto color =
          allocation ? allocation.getAs<IntegerAttr>("color") : IntegerAttr();
      auto value =
          valueId ? values.find(valueId.getValue().str()) : values.end();
      if (!allocation || !valueId || value == values.end() || !physicalType ||
          value->second.getDefiningOp<arith::ConstantOp>() ||
          physicalType.getValue() !=
              hostLiteralPhysicalType(value->second.getType()) ||
          !domain || domain.getValue() != "writable" || !kind ||
          kind.getValue() != "direct_color_candidate" || !color ||
          color.getInt() < 0 ||
          !directIds.insert(valueId.getValue().str()).second) {
        module.emitError("invalid Host direct color allocation inventory");
        signalPassFailure();
        return;
      }
      directPlans.push_back({allocation, value->second});
    }

    std::map<std::string, DictionaryAttr> siteById;
    for (Attribute attribute : sites) {
      auto site = dyn_cast<DictionaryAttr>(attribute);
      auto id = site ? site.getAs<StringAttr>("site_id") : StringAttr();
      if (!id || id.getValue().empty() ||
          !siteById.emplace(id.getValue().str(), site).second) {
        module.emitError("invalid Host literal materialization site inventory");
        signalPassFailure();
        return;
      }
    }
    std::map<std::string, DictionaryAttr> instanceById;
    for (Attribute attribute : instances) {
      auto instance = dyn_cast<DictionaryAttr>(attribute);
      auto id =
          instance ? instance.getAs<StringAttr>("instance_id") : StringAttr();
      if (!id || id.getValue().empty() ||
          !instanceById.emplace(id.getValue().str(), instance).second) {
        module.emitError(
            "invalid Host literal materialization instance inventory");
        signalPassFailure();
        return;
      }
    }
    std::map<std::string, DictionaryAttr> scratchById;
    for (Attribute attribute : scratchAllocations) {
      auto allocation = dyn_cast<DictionaryAttr>(attribute);
      auto id = allocation ? allocation.getAs<StringAttr>("instance_id")
                           : StringAttr();
      if (!id || id.getValue().empty() ||
          !scratchById.emplace(id.getValue().str(), allocation).second) {
        module.emitError("invalid Host literal scratch allocation inventory");
        signalPassFailure();
        return;
      }
    }
    std::map<std::string, DictionaryAttr> residentByKey;
    for (Attribute attribute : residentAllocations) {
      auto allocation = dyn_cast<DictionaryAttr>(attribute);
      auto key =
          allocation ? allocation.getAs<StringAttr>("key_id") : StringAttr();
      if (!key || key.getValue().empty() ||
          !residentByKey.emplace(key.getValue().str(), allocation).second) {
        module.emitError("invalid Host resident literal allocation inventory");
        signalPassFailure();
        return;
      }
    }
    std::map<std::string, int64_t> extentByType;
    for (Attribute attribute : writableExtents) {
      auto extent = dyn_cast<DictionaryAttr>(attribute);
      auto domain =
          extent ? extent.getAs<StringAttr>("storage_domain") : StringAttr();
      auto physicalType =
          extent ? extent.getAs<StringAttr>("physical_type") : StringAttr();
      auto count = extent ? extent.getAs<IntegerAttr>("extent") : IntegerAttr();
      if (!domain || domain.getValue() != "writable" || !physicalType ||
          physicalType.getValue().empty() || !count || count.getInt() < 0 ||
          !extentByType.emplace(physicalType.getValue().str(), count.getInt())
               .second) {
        module.emitError("invalid Host writable extent inventory");
        signalPassFailure();
        return;
      }
    }

    OpBuilder builder(module.getContext());
    struct RewritePlan {
      Operation *consumer;
      SmallVector<unsigned> operandIndices;
      Attribute literal;
      Type resultType;
      StringAttr literalKey;
      StringAttr instanceId;
      StringAttr recipe;
      StringAttr domain;
      StringAttr physicalType;
      IntegerAttr color;
      ArrayAttr seedBindings;
      ArrayAttr requiredExterns;
      IntegerAttr helperScratch;
      IntegerAttr generatedVersion;
      SmallVector<Operation *> originalConstants;
    };
    SmallVector<RewritePlan> plans;
    std::set<std::string> selectedInstances;
    std::string previousInstance;
    for (Attribute attribute : selections) {
      auto selection = dyn_cast<DictionaryAttr>(attribute);
      auto instanceId =
          selection ? selection.getAs<StringAttr>("instance_id") : StringAttr();
      auto placement =
          selection ? selection.getAs<StringAttr>("placement") : StringAttr();
      auto selectedSites =
          selection ? selection.getAs<ArrayAttr>("site_ids") : ArrayAttr();
      auto selectedLiteralKey =
          selection ? selection.getAs<StringAttr>("literal_key") : StringAttr();
      auto selectedLogicalType =
          selection ? selection.getAs<StringAttr>("logical_type")
                    : StringAttr();
      auto recipe =
          selection ? selection.getAs<StringAttr>("recipe") : StringAttr();
      auto domain = selection ? selection.getAs<StringAttr>("storage_domain")
                              : StringAttr();
      auto physicalType = selection
                              ? selection.getAs<StringAttr>("physical_type")
                              : StringAttr();
      auto color =
          selection ? selection.getAs<IntegerAttr>("color") : IntegerAttr();
      auto seedBindings =
          selection ? selection.getAs<ArrayAttr>("seed_bindings") : ArrayAttr();
      auto requiredExterns =
          selection ? selection.getAs<ArrayAttr>("required_externs")
                    : ArrayAttr();
      auto helperScratch = selection
                               ? selection.getAs<IntegerAttr>("helper_scratch")
                               : IntegerAttr();
      auto generatedVersion =
          selection ? selection.getAs<IntegerAttr>("generated_version")
                    : IntegerAttr();
      if (!selection || !instanceId || instanceId.getValue().empty() ||
          !placement || placement.getValue() != "rematerialized" ||
          !selectedSites || selectedSites.empty() || !selectedLiteralKey ||
          !selectedLogicalType || !recipe ||
          recipe.getValue() != "rematerialize_i32" || !domain ||
          domain.getValue() != "writable" || !physicalType ||
          physicalType.getValue() != "SystemInt32" || !color ||
          color.getInt() < 0 || !seedBindings || !requiredExterns ||
          !helperScratch || helperScratch.getInt() != 0 || !generatedVersion ||
          generatedVersion.getInt() != 1 ||
          !selectedInstances.insert(instanceId.getValue().str()).second ||
          (!previousInstance.empty() &&
           previousInstance >= instanceId.getValue().str())) {
        module.emitError("invalid Host literal materialization selection");
        signalPassFailure();
        return;
      }
      previousInstance = instanceId.getValue().str();

      auto foundInstance = instanceById.find(instanceId.getValue().str());
      auto foundScratch = scratchById.find(instanceId.getValue().str());
      if (foundInstance == instanceById.end() ||
          foundScratch == scratchById.end()) {
        module.emitError(
            "Host literal selection references an unknown instance");
        signalPassFailure();
        return;
      }
      DictionaryAttr instance = foundInstance->second;
      DictionaryAttr scratch = foundScratch->second;
      auto consumerId = instance.getAs<StringAttr>("consumer_id");
      auto literalKey = instance.getAs<StringAttr>("literal_key");
      auto logicalType = instance.getAs<StringAttr>("logical_type");
      auto instancePhysical = instance.getAs<StringAttr>("physical_type");
      auto instanceSites = instance.getAs<ArrayAttr>("site_ids");
      auto scratchColor = scratch.getAs<IntegerAttr>("color");
      auto scratchDomain = scratch.getAs<StringAttr>("storage_domain");
      auto scratchAllocation = scratch.getAs<StringAttr>("allocation");
      if (!consumerId || !literalKey || !logicalType || !instancePhysical ||
          selectedLiteralKey != literalKey ||
          selectedLogicalType != logicalType ||
          instancePhysical != physicalType || !instanceSites ||
          instanceSites.empty() || selectedSites != instanceSites ||
          !scratchColor || scratchColor.getInt() != color.getInt() ||
          !scratchDomain || scratchDomain.getValue() != "writable" ||
          !scratchAllocation ||
          scratchAllocation.getValue() != "immediate_consumer_candidate") {
        module.emitError("Host literal selection disagrees with allocation");
        signalPassFailure();
        return;
      }
      auto extent = extentByType.find(physicalType.getValue().str());
      if (extent == extentByType.end() || color.getInt() >= extent->second) {
        module.emitError("Host literal selection color exceeds its extent");
        signalPassFailure();
        return;
      }

      std::set<int64_t> seedColors;
      for (Attribute seedAttribute : seedBindings) {
        auto seed = dyn_cast<DictionaryAttr>(seedAttribute);
        auto key = seed ? seed.getAs<StringAttr>("literal_key") : StringAttr();
        auto seedType =
            seed ? seed.getAs<StringAttr>("physical_type") : StringAttr();
        auto seedDomain =
            seed ? seed.getAs<StringAttr>("storage_domain") : StringAttr();
        auto seedColor =
            seed ? seed.getAs<IntegerAttr>("color") : IntegerAttr();
        auto resident = key ? residentByKey.find(key.getValue().str())
                            : residentByKey.end();
        auto residentType =
            resident != residentByKey.end()
                ? resident->second.getAs<StringAttr>("physical_type")
                : StringAttr();
        auto residentColor = resident != residentByKey.end()
                                 ? resident->second.getAs<IntegerAttr>("color")
                                 : IntegerAttr();
        auto residentPlacement =
            resident != residentByKey.end()
                ? resident->second.getAs<StringAttr>("placement")
                : StringAttr();
        if (!seed || !key || !seedType || seedType != residentType ||
            seedType.getValue() != "SystemInt32" || !seedDomain ||
            seedDomain.getValue() != "resident" || !seedColor ||
            !residentColor || seedColor.getInt() != residentColor.getInt() ||
            !seedColors.insert(seedColor.getInt()).second ||
            !residentPlacement ||
            residentPlacement.getValue() != "resident_shared") {
          module.emitError(
              "Host literal selection has an invalid seed binding");
          signalPassFailure();
          return;
        }
      }

      Operation *consumer = nullptr;
      Attribute literal;
      Type resultType;
      SmallVector<unsigned> operandIndices;
      SmallVector<Operation *> originalConstants;
      for (Attribute siteAttribute : instanceSites) {
        auto siteId = dyn_cast<StringAttr>(siteAttribute);
        auto foundSite =
            siteId ? siteById.find(siteId.getValue().str()) : siteById.end();
        if (!siteId || foundSite == siteById.end()) {
          module.emitError("Host literal selection references an unknown site");
          signalPassFailure();
          return;
        }
        DictionaryAttr site = foundSite->second;
        auto siteInstance = site.getAs<StringAttr>("scratch_instance_id");
        auto siteConsumer = site.getAs<StringAttr>("consumer_id");
        auto operandIndex = site.getAs<IntegerAttr>("operand_index");
        auto eligible = site.getAs<BoolAttr>("scratch_eligible");
        auto operation = siteConsumer
                             ? operations.find(siteConsumer.getValue().str())
                             : operations.end();
        if (!siteInstance || siteInstance != instanceId || !siteConsumer ||
            siteConsumer != consumerId || !operandIndex ||
            operandIndex.getInt() < 0 || !eligible || !eligible.getValue() ||
            operation == operations.end() ||
            static_cast<uint64_t>(operandIndex.getInt()) >=
                operation->second->getNumOperands()) {
          module.emitError("Host literal selection has a stale site");
          signalPassFailure();
          return;
        }
        Value operand = operation->second->getOperand(operandIndex.getInt());
        auto constant = operand.getDefiningOp<arith::ConstantOp>();
        HostLiteralKey key = constant ? getHostLiteralKey(constant.getType(),
                                                          constant.getValue())
                                      : HostLiteralKey();
        if (!constant || key.id != literalKey.getValue() ||
            hostLiteralTypeText(constant.getType()) != logicalType.getValue() ||
            (consumer && consumer != operation->second) ||
            (literal && literal != constant.getValue()) ||
            (resultType && resultType != constant.getType())) {
          module.emitError(
              "Host literal selection no longer matches its operand");
          signalPassFailure();
          return;
        }
        consumer = operation->second;
        literal = constant.getValue();
        resultType = constant.getType();
        operandIndices.push_back(static_cast<unsigned>(operandIndex.getInt()));
        if (!llvm::is_contained(originalConstants, constant.getOperation()))
          originalConstants.push_back(constant.getOperation());
      }

      auto expectedRecipe = getHostLiteralI32Recipe(resultType, literal);
      if (!expectedRecipe ||
          seedBindings.size() != expectedRecipe->seedValues.size() ||
          requiredExterns.size() != expectedRecipe->requiredExterns.size()) {
        module.emitError("Host literal selection disagrees with its recipe");
        signalPassFailure();
        return;
      }
      for (auto [seedAttribute, expectedSeed] :
           llvm::zip(seedBindings, expectedRecipe->seedValues)) {
        auto seed = dyn_cast<DictionaryAttr>(seedAttribute);
        auto key = seed ? seed.getAs<StringAttr>("literal_key") : StringAttr();
        Type i32 = builder.getI32Type();
        HostLiteralKey expectedKey =
            getHostLiteralKey(i32, builder.getIntegerAttr(i32, expectedSeed));
        if (!key || key.getValue() != expectedKey.id) {
          module.emitError("Host literal selection disagrees with its recipe");
          signalPassFailure();
          return;
        }
      }
      for (auto [externAttribute, expectedExtern] :
           llvm::zip(requiredExterns, expectedRecipe->requiredExterns)) {
        auto name = dyn_cast<StringAttr>(externAttribute);
        if (!name || name.getValue() != expectedExtern) {
          module.emitError("Host literal selection disagrees with its recipe");
          signalPassFailure();
          return;
        }
      }

      plans.push_back({consumer, operandIndices, literal, resultType,
                       literalKey, instanceId, recipe, domain, physicalType,
                       color, seedBindings, requiredExterns, helperScratch,
                       generatedVersion, originalConstants});
    }

    SmallVector<Attribute> materializedInstances;
    for (const RewritePlan &plan : plans) {
      builder.setInsertionPoint(plan.consumer);
      OperationState state(plan.consumer->getLoc(),
                           udonhl::LiteralMaterializeOp::getOperationName());
      state.addTypes(plan.resultType);
      state.addAttribute("literal", plan.literal);
      state.addAttribute("literal_key", plan.literalKey);
      state.addAttribute("instance_id", plan.instanceId);
      state.addAttribute("recipe", plan.recipe);
      state.addAttribute("storage_domain", plan.domain);
      state.addAttribute("physical_type", plan.physicalType);
      state.addAttribute("color", plan.color);
      state.addAttribute("seed_bindings", plan.seedBindings);
      state.addAttribute("required_externs", plan.requiredExterns);
      state.addAttribute("helper_scratch", plan.helperScratch);
      state.addAttribute("generated_version", plan.generatedVersion);
      Operation *materialized = builder.create(state);
      for (unsigned operandIndex : plan.operandIndices)
        plan.consumer->setOperand(operandIndex, materialized->getResult(0));
      materializedInstances.push_back(plan.instanceId);
    }
    std::set<Operation *> originalConstants;
    for (const RewritePlan &plan : plans)
      originalConstants.insert(plan.originalConstants.begin(),
                               plan.originalConstants.end());
    for (Operation *constant : originalConstants)
      if (constant->use_empty())
        constant->erase();

    llvm::DenseMap<Value, std::string> rewrittenValueIds;
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      std::function<void(Region &, const std::string &)> collect;
      collect = [&](Region &region, const std::string &regionPath) {
        for (auto [blockIndex, block] : llvm::enumerate(region.getBlocks())) {
          std::string blockPath =
              regionPath + ":block:" + std::to_string(blockIndex);
          for (auto [argumentIndex, argument] :
               llvm::enumerate(block.getArguments()))
            rewrittenValueIds.try_emplace(
                argument,
                blockPath + ":argument:" + std::to_string(argumentIndex));
          for (auto [operationIndex, operation] : llvm::enumerate(block)) {
            std::string operationPath =
                blockPath + ":operation:" + std::to_string(operationIndex);
            for (auto [resultIndex, result] :
                 llvm::enumerate(operation.getResults()))
              rewrittenValueIds.try_emplace(
                  result,
                  operationPath + ":result:" + std::to_string(resultIndex));
            for (auto [regionIndex, nested] :
                 llvm::enumerate(operation.getRegions()))
              collect(nested,
                      operationPath + ":region:" + std::to_string(regionIndex));
          }
        }
      };
      collect(function.getBody(),
              "function:" + function.getSymName().str() + ":region:0");
    }
    SmallVector<Attribute> rewrittenDirectAllocations;
    for (const auto &[source, value] : directPlans) {
      NamedAttrList fields(source);
      fields.set("value_id",
                 builder.getStringAttr(rewrittenValueIds.lookup(value)));
      rewrittenDirectAllocations.push_back(builder.getDictionaryAttr(fields));
    }
    module->setAttr("vrc.host_direct_color_allocations",
                    builder.getArrayAttr(rewrittenDirectAllocations));

    module->setAttr("vrc.host_literal_materialization_version",
                    builder.getI32IntegerAttr(1));
    module->setAttr("vrc.host_literal_materialized_instances",
                    builder.getArrayAttr(materializedInstances));
  }
};

} // namespace
} // namespace vrc
