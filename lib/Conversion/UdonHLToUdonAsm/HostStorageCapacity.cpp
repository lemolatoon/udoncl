#include "VRC/Conversion/HostStorageCapacity.h"
#include "VRC/Conversion/Passes.h"
#include "VRC/Support/UdonLimits.h"
#include "HostStorageCapacityInternal.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_MEASUREHOSTSTORAGECAPACITYPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {

FailureOr<uint32_t> readCount(DictionaryAttr measurement, StringRef name) {
  auto value = measurement.getAs<IntegerAttr>(name);
  if (!value || !value.getType().isInteger(64) || value.getInt() < 0 ||
      static_cast<uint64_t>(value.getInt()) > UINT32_MAX)
    return failure();
  return static_cast<uint32_t>(value.getInt());
}

DictionaryAttr measurementAttr(MLIRContext *context,
                               const HostStorageCapacityMeasurement &value) {
  Builder builder(context);
  return builder.getDictionaryAttr(
      {builder.getNamedAttr("schema_version", builder.getI32IntegerAttr(1)),
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
       builder.getNamedAttr("fits", builder.getBoolAttr(value.fits()))});
}

struct MeasureHostStorageCapacityPass
    : impl::MeasureHostStorageCapacityPassBase<MeasureHostStorageCapacityPass> {
  void runOnOperation() override {
    auto measurement = measureHostStorageCapacity(getOperation());
    if (failed(measurement)) {
      signalPassFailure();
      return;
    }
    getOperation()->setAttr(
        "vrc.host_storage_capacity_measurement",
        measurementAttr(getOperation().getContext(), *measurement));
  }
};

} // namespace

FailureOr<HostStorageCapacityMeasurement>
measureHostStorageCapacity(ModuleOp module) {
  OwningOpRef<ModuleOp> clone = cast<ModuleOp>(module->clone());
  PassManager manager(module.getContext());
  manager.addPass(createHostStorageMeasurementLoweringPass());
  if (failed(manager.run(*clone)))
    return failure();
  auto measurement = (*clone)->getAttrOfType<DictionaryAttr>(
      "vrc.host_storage_capacity_measurement");
  auto schema = measurement ? measurement.getAs<IntegerAttr>("schema_version")
                            : IntegerAttr();
  auto fits = measurement ? measurement.getAs<BoolAttr>("fits") : BoolAttr();
  if (!schema || !schema.getType().isInteger(32) || schema.getInt() != 1 ||
      !fits)
    return failure();
  auto addresses = readCount(measurement, "address_count");
  auto dataSlots = readCount(measurement, "data_slot_count");
  auto externs = readCount(measurement, "unique_extern_count");
  auto addressLimit = readCount(measurement, "address_limit");
  auto dataCapacity = readCount(measurement, "data_slot_capacity");
  auto reserve = readCount(measurement, "final_data_slot_reserve");
  auto required = readCount(measurement, "required_reduction");
  if (failed(addresses) || failed(dataSlots) || failed(externs) ||
      failed(addressLimit) || failed(dataCapacity) || failed(reserve) ||
      failed(required) || *dataSlots + *externs != *addresses ||
      *addressLimit + *reserve != kUdonHeapMaximumCapacity ||
      fits.getValue() != (*required == 0))
    return failure();
  return HostStorageCapacityMeasurement{*addresses,    *dataSlots,    *externs,
                                        *addressLimit, *dataCapacity, *reserve,
                                        *required};
}

} // namespace vrc
