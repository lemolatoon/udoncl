#include "VRC/Conversion/HostStorageCapacity.h"
#include "VRC/Conversion/Passes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include <algorithm>

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_VERIFYHOSTSTORAGECAPACITYPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {

FailureOr<uint32_t> readCount(DictionaryAttr report, StringRef name,
                              unsigned width) {
  auto value = report.getAs<IntegerAttr>(name);
  if (!value || value.getType().getIntOrFloatBitWidth() != width ||
      value.getInt() < 0 || static_cast<uint64_t>(value.getInt()) > UINT32_MAX)
    return failure();
  return static_cast<uint32_t>(value.getInt());
}

struct FitReport {
  HostStorageCapacityMeasurement baseline;
  HostStorageCapacityMeasurement final;
  uint32_t attempts;
  uint32_t requestedSavings;
};

FailureOr<FitReport> parseFitReport(ModuleOp module) {
  auto report =
      module->getAttrOfType<DictionaryAttr>("vrc.host_storage_capacity_fit");
  auto schema = report ? readCount(report, "schema_version", 32)
                       : FailureOr<uint32_t>(failure());
  if (!report || report.size() != 17 || failed(schema) || *schema != 2)
    return failure();
  auto read = [&](StringRef name) { return readCount(report, name, 64); };
  auto ba = read("baseline_address_count");
  auto bd = read("baseline_data_slot_count");
  auto be = read("baseline_unique_extern_count");
  auto bl = read("baseline_address_limit");
  auto bc = read("baseline_data_slot_capacity");
  auto br = read("baseline_final_data_slot_reserve");
  auto bx = read("baseline_required_reduction");
  auto fa = read("final_address_count");
  auto fd = read("final_data_slot_count");
  auto fe = read("final_unique_extern_count");
  auto fl = read("final_address_limit");
  auto fc = read("final_data_slot_capacity");
  auto fr = read("final_data_slot_reserve");
  auto fx = read("final_required_reduction");
  auto attempts = readCount(report, "attempt_count", 32);
  auto requested = readCount(report, "requested_resident_slot_savings", 32);
  if (failed(ba) || failed(bd) || failed(be) || failed(bl) || failed(bc) ||
      failed(br) || failed(bx) || failed(fa) || failed(fd) || failed(fe) ||
      failed(fl) || failed(fc) || failed(fr) || failed(fx) ||
      failed(attempts) || failed(requested))
    return failure();
  HostStorageCapacityMeasurement baseline{*ba, *bd, *be, *bl, *bc, *br, *bx};
  HostStorageCapacityMeasurement final{*fa, *fd, *fe, *fl, *fc, *fr, *fx};
  auto expectedReduction = [](const HostStorageCapacityMeasurement &value) {
    uint64_t addressOverflow =
        value.addressCount > value.addressLimit
            ? uint64_t(value.addressCount) - value.addressLimit
            : 0;
    uint64_t dataOverflow =
        value.dataSlotCount > value.dataSlotCapacity
            ? uint64_t(value.dataSlotCount) - value.dataSlotCapacity
            : 0;
    return static_cast<uint32_t>(std::max(addressOverflow, dataOverflow));
  };
  if (uint64_t(baseline.dataSlotCount) + baseline.uniqueExternCount !=
          baseline.addressCount ||
      uint64_t(final.dataSlotCount) + final.uniqueExternCount !=
          final.addressCount ||
      uint64_t(baseline.addressLimit) + baseline.finalDataSlotReserve != 512 ||
      uint64_t(final.addressLimit) + final.finalDataSlotReserve != 512 ||
      baseline.addressLimit != final.addressLimit ||
      baseline.dataSlotCapacity != final.dataSlotCapacity ||
      baseline.finalDataSlotReserve != final.finalDataSlotReserve ||
      baseline.requiredReduction != expectedReduction(baseline) ||
      final.requiredReduction != expectedReduction(final) || !final.fits())
    return failure();
  if (*attempts == 0) {
    if (*requested != 0 || baseline.addressCount != final.addressCount ||
        baseline.dataSlotCount != final.dataSlotCount ||
        baseline.uniqueExternCount != final.uniqueExternCount ||
        baseline.addressLimit != final.addressLimit ||
        baseline.dataSlotCapacity != final.dataSlotCapacity ||
        baseline.finalDataSlotReserve != final.finalDataSlotReserve ||
        baseline.requiredReduction != final.requiredReduction)
      return failure();
    if (module->hasAttr("vrc.host_storage_pressure_request") ||
        module->hasAttr("vrc.host_literal_selected_placement_version") ||
        module->hasAttr("vrc.host_literal_selected_placements") ||
        module->hasAttr("vrc.host_storage_selection") ||
        module->hasAttr("vrc.host_literal_materialization_version") ||
        module->hasAttr("vrc.host_literal_materialized_instances"))
      return failure();
  } else if (*requested == 0 || baseline.requiredReduction == 0 ||
             *requested < baseline.requiredReduction ||
             *attempts != *requested - baseline.requiredReduction + 1) {
    return failure();
  } else {
    auto request = module->getAttrOfType<DictionaryAttr>(
        "vrc.host_storage_pressure_request");
    auto requestSchema = request ? readCount(request, "schema_version", 32)
                                 : FailureOr<uint32_t>(failure());
    auto measured = request ? readCount(request, "measured_data_slots", 32)
                            : FailureOr<uint32_t>(failure());
    auto capacity = request ? readCount(request, "data_slot_capacity", 32)
                            : FailureOr<uint32_t>(failure());
    auto required =
        request ? readCount(request, "required_resident_slot_savings", 32)
                : FailureOr<uint32_t>(failure());
    auto selectedVersion = module->getAttrOfType<IntegerAttr>(
        "vrc.host_literal_selected_placement_version");
    auto selections = module->getAttrOfType<ArrayAttr>(
        "vrc.host_literal_selected_placements");
    auto selection =
        module->getAttrOfType<DictionaryAttr>("vrc.host_storage_selection");
    auto selectionSchema = selection
                               ? readCount(selection, "schema_version", 32)
                               : FailureOr<uint32_t>(failure());
    auto selectionRequired =
        selection ? readCount(selection, "required_resident_slot_savings", 32)
                  : FailureOr<uint32_t>(failure());
    auto selectionGuaranteed =
        selection ? readCount(selection, "guaranteed_resident_slot_savings", 32)
                  : FailureOr<uint32_t>(failure());
    auto selectionKeys =
        selection ? readCount(selection, "selected_literal_key_count", 32)
                  : FailureOr<uint32_t>(failure());
    auto materializedVersion = module->getAttrOfType<IntegerAttr>(
        "vrc.host_literal_materialization_version");
    auto materialized = module->getAttrOfType<ArrayAttr>(
        "vrc.host_literal_materialized_instances");
    if (!request || request.size() != 4 || failed(requestSchema) ||
        *requestSchema != 1 || failed(measured) || failed(capacity) ||
        failed(required) || *measured != baseline.dataSlotCount ||
        *required != *requested || *required >= *measured || *capacity < 1 ||
        *capacity > 512 || *capacity != *measured - *required ||
        !selectedVersion || !selectedVersion.getType().isInteger(32) ||
        selectedVersion.getInt() != 1 || !selections || selections.empty() ||
        !selection || selection.size() != 4 || failed(selectionSchema) ||
        *selectionSchema != 1 || failed(selectionRequired) ||
        failed(selectionGuaranteed) || failed(selectionKeys) ||
        *selectionRequired != *requested ||
        *selectionGuaranteed != *requested || *selectionKeys != *requested ||
        !materializedVersion || !materializedVersion.getType().isInteger(32) ||
        materializedVersion.getInt() != 1 || !materialized ||
        materialized.size() != selections.size())
      return failure();
  }
  return FitReport{baseline, final, *attempts, *requested};
}

bool equalMeasurement(const HostStorageCapacityMeasurement &left,
                      const HostStorageCapacityMeasurement &right) {
  return left.addressCount == right.addressCount &&
         left.dataSlotCount == right.dataSlotCount &&
         left.uniqueExternCount == right.uniqueExternCount &&
         left.addressLimit == right.addressLimit &&
         left.dataSlotCapacity == right.dataSlotCapacity &&
         left.finalDataSlotReserve == right.finalDataSlotReserve &&
         left.requiredReduction == right.requiredReduction;
}

DictionaryAttr verificationAttr(MLIRContext *context, const FitReport &fit) {
  Builder builder(context);
  const auto &value = fit.final;
  return builder.getDictionaryAttr({
      builder.getNamedAttr("schema_version", builder.getI32IntegerAttr(1)),
      builder.getNamedAttr("fit_schema_version", builder.getI32IntegerAttr(2)),
      builder.getNamedAttr("address_count",
                           builder.getI64IntegerAttr(value.addressCount)),
      builder.getNamedAttr("data_slot_count",
                           builder.getI64IntegerAttr(value.dataSlotCount)),
      builder.getNamedAttr("unique_extern_count",
                           builder.getI64IntegerAttr(value.uniqueExternCount)),
      builder.getNamedAttr("address_limit",
                           builder.getI64IntegerAttr(value.addressLimit)),
      builder.getNamedAttr("data_slot_capacity",
                           builder.getI64IntegerAttr(value.dataSlotCapacity)),
      builder.getNamedAttr(
          "final_data_slot_reserve",
          builder.getI64IntegerAttr(value.finalDataSlotReserve)),
      builder.getNamedAttr("required_reduction",
                           builder.getI64IntegerAttr(value.requiredReduction)),
      builder.getNamedAttr("fits", builder.getBoolAttr(value.fits())),
      builder.getNamedAttr("attempt_count",
                           builder.getI32IntegerAttr(fit.attempts)),
      builder.getNamedAttr("requested_resident_slot_savings",
                           builder.getI32IntegerAttr(fit.requestedSavings)),
  });
}

struct VerifyHostStorageCapacityPass
    : impl::VerifyHostStorageCapacityPassBase<VerifyHostStorageCapacityPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto fit = parseFitReport(module);
    if (failed(fit)) {
      module.emitError("invalid Host storage capacity fit report");
      signalPassFailure();
      return;
    }
    auto measured = measureHostStorageCapacity(module);
    if (failed(measured)) {
      module.emitError("failed to measure fitted Host storage");
      signalPassFailure();
      return;
    }
    if (!measured->fits() || !equalMeasurement(*measured, fit->final)) {
      module.emitError()
          << "fitted Host storage does not match the exact final Udon address "
             "plan; expected "
             "address/data/extern/limit/capacity/reserve/required="
          << fit->final.addressCount << "/" << fit->final.dataSlotCount << "/"
          << fit->final.uniqueExternCount << "/" << fit->final.addressLimit
          << "/" << fit->final.dataSlotCapacity << "/"
          << fit->final.finalDataSlotReserve << "/"
          << fit->final.requiredReduction
          << ", actual=" << measured->addressCount << "/"
          << measured->dataSlotCount << "/" << measured->uniqueExternCount
          << "/" << measured->addressLimit << "/" << measured->dataSlotCapacity
          << "/" << measured->finalDataSlotReserve << "/"
          << measured->requiredReduction << "; rerun capacity fitting";
      signalPassFailure();
      return;
    }
    DictionaryAttr expected = verificationAttr(module.getContext(), *fit);
    if (Attribute existing =
            module->getAttr("vrc.host_storage_capacity_verification")) {
      if (existing != expected) {
        module.emitError("stale Host storage capacity verification report");
        signalPassFailure();
      }
      return;
    }
    module->setAttr("vrc.host_storage_capacity_verification", expected);
  }
};

} // namespace
} // namespace vrc
