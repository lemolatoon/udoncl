#ifndef UDONCL_CONVERSION_ONNX_TO_UDON_HOST_H
#define UDONCL_CONVERSION_ONNX_TO_UDON_HOST_H

#include "mlir/Support/LogicalResult.h"

namespace mlir {
class ModuleOp;
}

namespace vrc {
mlir::LogicalResult lowerONNXToUdonHost(mlir::ModuleOp module);
} // namespace vrc

#endif
