#ifndef VRC_CONVERSION_UDONHLTOUDONASM_HOSTSTORAGECAPACITYINTERNAL_H
#define VRC_CONVERSION_UDONHLTOUDONASM_HOSTSTORAGECAPACITYINTERNAL_H

#include <memory>

namespace mlir {
class Pass;
}

namespace vrc {

std::unique_ptr<mlir::Pass> createHostStorageMeasurementLoweringPass();

} // namespace vrc

#endif
