#include "VRC/Conversion/HostStorageCapacity.h"
#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "VRC/Support/UdonLimits.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/PassManager.h"
#include <limits>

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_FITHOSTSTORAGECAPACITYPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {

bool hasI32Version(ModuleOp module, StringRef name, int64_t expected) {
  auto value = module->getAttrOfType<IntegerAttr>(name);
  return value && value.getType().isInteger(32) && value.getInt() == expected;
}

LogicalResult prepareCandidate(ModuleOp module) {
  PassManager manager(module.getContext());
  manager.addPass(createAllocateHostLiteralStoragePass());
  manager.addPass(createMaterializeHostStoragePass());
  return manager.run(module);
}

void setPressureRequest(ModuleOp module, uint32_t measuredDataSlots,
                        uint32_t requestedSavings) {
  Builder builder(module.getContext());
  module->setAttr(
      "vrc.host_storage_pressure_request",
      builder.getDictionaryAttr({
          builder.getNamedAttr("schema_version", builder.getI32IntegerAttr(1)),
          builder.getNamedAttr("measured_data_slots",
                               builder.getI32IntegerAttr(measuredDataSlots)),
          builder.getNamedAttr(
              "data_slot_capacity",
              builder.getI32IntegerAttr(measuredDataSlots - requestedSavings)),
          builder.getNamedAttr("required_resident_slot_savings",
                               builder.getI32IntegerAttr(requestedSavings)),
      }));
}

void setFitReport(ModuleOp module,
                  const HostStorageCapacityMeasurement &baseline,
                  const HostStorageCapacityMeasurement &final,
                  uint32_t attempts, uint32_t requestedSavings) {
  Builder builder(module.getContext());
  module->setAttr(
      "vrc.host_storage_capacity_fit",
      builder.getDictionaryAttr({
          builder.getNamedAttr("schema_version", builder.getI32IntegerAttr(2)),
          builder.getNamedAttr(
              "baseline_address_count",
              builder.getI64IntegerAttr(baseline.addressCount)),
          builder.getNamedAttr(
              "baseline_data_slot_count",
              builder.getI64IntegerAttr(baseline.dataSlotCount)),
          builder.getNamedAttr(
              "baseline_unique_extern_count",
              builder.getI64IntegerAttr(baseline.uniqueExternCount)),
          builder.getNamedAttr(
              "baseline_address_limit",
              builder.getI64IntegerAttr(baseline.addressLimit)),
          builder.getNamedAttr(
              "baseline_data_slot_capacity",
              builder.getI64IntegerAttr(baseline.dataSlotCapacity)),
          builder.getNamedAttr(
              "baseline_final_data_slot_reserve",
              builder.getI64IntegerAttr(baseline.finalDataSlotReserve)),
          builder.getNamedAttr(
              "baseline_required_reduction",
              builder.getI64IntegerAttr(baseline.requiredReduction)),
          builder.getNamedAttr("final_address_count",
                               builder.getI64IntegerAttr(final.addressCount)),
          builder.getNamedAttr("final_data_slot_count",
                               builder.getI64IntegerAttr(final.dataSlotCount)),
          builder.getNamedAttr(
              "final_unique_extern_count",
              builder.getI64IntegerAttr(final.uniqueExternCount)),
          builder.getNamedAttr("final_address_limit",
                               builder.getI64IntegerAttr(final.addressLimit)),
          builder.getNamedAttr(
              "final_data_slot_capacity",
              builder.getI64IntegerAttr(final.dataSlotCapacity)),
          builder.getNamedAttr(
              "final_data_slot_reserve",
              builder.getI64IntegerAttr(final.finalDataSlotReserve)),
          builder.getNamedAttr(
              "final_required_reduction",
              builder.getI64IntegerAttr(final.requiredReduction)),
          builder.getNamedAttr("attempt_count",
                               builder.getI32IntegerAttr(attempts)),
          builder.getNamedAttr("requested_resident_slot_savings",
                               builder.getI32IntegerAttr(requestedSavings)),
      }));
}

void commitCandidate(ModuleOp destination, ModuleOp source) {
  destination->setAttrs(source->getAttrDictionary());
  destination.getBodyRegion().takeBody(source.getBodyRegion());
}

struct FitHostStorageCapacityPass
    : impl::FitHostStorageCapacityPassBase<FitHostStorageCapacityPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (!hasI32Version(module, "vrc.host_backend_literal_use_version", 1) ||
        !hasI32Version(module, "vrc.host_literal_inventory_version", 1) ||
        !hasI32Version(module, "vrc.host_literal_use_site_version", 3) ||
        module->hasAttr("vrc.host_storage_capacity_fit") ||
        module->hasAttr("vrc.host_storage_capacity_measurement") ||
        module->hasAttr("vrc.host_storage_pressure_request") ||
        module->hasAttr("vrc.host_literal_selected_placement_version") ||
        module->hasAttr("vrc.host_literal_selected_placements") ||
        module->hasAttr("vrc.host_storage_selection") ||
        module->hasAttr("vrc.host_literal_materialization_version") ||
        module->hasAttr("vrc.host_literal_materialized_instances")) {
      module.emitError("Host storage capacity fitting requires pristine "
                       "post-discovery IR");
      signalPassFailure();
      return;
    }

    OwningOpRef<ModuleOp> pristine = cast<ModuleOp>(module->clone());
    OwningOpRef<ModuleOp> candidate = cast<ModuleOp>(pristine->clone());
    if (failed(prepareCandidate(*candidate))) {
      signalPassFailure();
      return;
    }
    auto baseline = measureHostStorageCapacity(*candidate);
    if (failed(baseline)) {
      module.emitError("failed to measure candidate Host storage capacity");
      signalPassFailure();
      return;
    }
    if (baseline->fits()) {
      setFitReport(*candidate, *baseline, *baseline, 0, 0);
      commitCandidate(module, *candidate);
      return;
    }

    uint32_t requestedSavings = baseline->requiredReduction;
    if (baseline->dataSlotCount >
            static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) ||
        requestedSavings == 0 || requestedSavings >= baseline->dataSlotCount ||
        baseline->dataSlotCount - requestedSavings > kUdonHeapMaximumCapacity) {
      module.emitError("measured Host storage pressure cannot be represented "
                       "by a version 1 selection request");
      signalPassFailure();
      return;
    }
    uint32_t attempts = 0;
    while (requestedSavings < baseline->dataSlotCount) {
      ++attempts;
      OwningOpRef<ModuleOp> attempt = cast<ModuleOp>(pristine->clone());
      setPressureRequest(*attempt, baseline->dataSlotCount, requestedSavings);
      if (failed(prepareCandidate(*attempt))) {
        module.emitError("failed to prepare Host storage selection attempt");
        signalPassFailure();
        return;
      }
      auto measurement = measureHostStorageCapacity(*attempt);
      if (failed(measurement)) {
        module.emitError("failed to remeasure selected Host storage capacity");
        signalPassFailure();
        return;
      }
      if (measurement->fits()) {
        setFitReport(*attempt, *baseline, *measurement, attempts,
                     requestedSavings);
        commitCandidate(module, *attempt);
        return;
      }
      ++requestedSavings;
    }

    module.emitError("Host storage capacity cannot fit even after selecting "
                     "every data slot");
    signalPassFailure();
  }
};

} // namespace
} // namespace vrc
