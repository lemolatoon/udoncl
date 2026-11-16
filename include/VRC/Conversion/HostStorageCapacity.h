#ifndef VRC_CONVERSION_HOSTSTORAGECAPACITY_H
#define VRC_CONVERSION_HOSTSTORAGECAPACITY_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include <cstdint>
#include <string>

namespace vrc {

struct HostStorageCapacityMeasurement {
  uint32_t addressCount;
  uint32_t dataSlotCount;
  uint32_t uniqueExternCount;
  uint32_t addressLimit;
  uint32_t dataSlotCapacity;
  uint32_t finalDataSlotReserve;
  uint32_t requiredReduction;
  mlir::DictionaryAttr loweringPlan;

  bool fits() const { return requiredReduction == 0; }
};

mlir::FailureOr<HostStorageCapacityMeasurement>
measureHostStorageCapacity(mlir::ModuleOp module);

std::string hostStorageIRFingerprint(mlir::ModuleOp module);

} // namespace vrc

#endif
