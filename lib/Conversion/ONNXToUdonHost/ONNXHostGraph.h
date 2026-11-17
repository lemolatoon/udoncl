#ifndef VRC_CONVERSION_ONNXTOUDONHOST_ONNXHOSTGRAPH_H
#define VRC_CONVERSION_ONNXTOUDONHOST_ONNXHOSTGRAPH_H

#include "mlir/Support/LogicalResult.h"

namespace mlir {
class ModuleOp;
}

namespace vrc {
mlir::LogicalResult validateONNXHostGraph(mlir::ModuleOp module);
mlir::LogicalResult planONNXHostGraph(mlir::ModuleOp module);
} // namespace vrc

#endif
