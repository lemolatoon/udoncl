#ifndef VRC_CONVERSION_ONNXTOUDONHOST_ONNXDIMENSIONEXPRS_H
#define VRC_CONVERSION_ONNXTOUDONHOST_ONNXDIMENSIONEXPRS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"

#include <cstdint>

namespace vrc {
/// Module attribute: dictionary from every bounded ONNX dimension name to its
/// inclusive upper bound. A bounded dimension takes any extent in
/// [0, bound] at run time; DIMENSIONS fixes the other model dimensions.
inline constexpr llvm::StringLiteral onnxDimensionBoundsAttr =
    "udoncl.dimension_bounds";
/// Argument attribute of a graph input with a bounded dynamic axis: an array
/// of {axis, symbol, offset} dictionaries; the extent of `axis` is
/// `symbol + offset`.
inline constexpr llvm::StringLiteral onnxDimensionExprsAttr =
    "udoncl.dim_exprs";

inline mlir::DictionaryAttr
makeONNXDimensionExprAttr(mlir::MLIRContext *context, int64_t axis,
                          llvm::StringRef symbol, int64_t offset) {
  mlir::Builder builder(context);
  return builder.getDictionaryAttr(
      {builder.getNamedAttr("axis", builder.getI64IntegerAttr(axis)),
       builder.getNamedAttr("offset", builder.getI64IntegerAttr(offset)),
       builder.getNamedAttr("symbol", builder.getStringAttr(symbol))});
}
} // namespace vrc

#endif
