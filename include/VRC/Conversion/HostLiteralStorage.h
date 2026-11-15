#ifndef VRC_CONVERSION_HOSTLITERALSTORAGE_H
#define VRC_CONVERSION_HOSTLITERALSTORAGE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace vrc {

mlir::LogicalResult allocateHostLiteralStorage(mlir::ModuleOp module);

} // namespace vrc

#endif
