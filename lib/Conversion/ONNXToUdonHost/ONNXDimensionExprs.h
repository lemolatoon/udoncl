#ifndef VRC_CONVERSION_ONNXTOUDONHOST_ONNXDIMENSIONEXPRS_H
#define VRC_CONVERSION_ONNXTOUDONHOST_ONNXDIMENSIONEXPRS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <string>

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

/// An integer of an ONNX graph whose value may depend on one bounded
/// dimension: a constant, `symbol + offset`, or unknown.
struct DimExpr {
  mlir::StringAttr symbol;
  int64_t offset = 0;
  bool known = true;

  static DimExpr constant(int64_t value) { return {{}, value, true}; }
  static DimExpr of(mlir::StringAttr symbol, int64_t offset) {
    return {symbol, offset, true};
  }
  static DimExpr unknown() { return {{}, 0, false}; }
  bool isConstant() const { return known && !symbol; }
  bool isSymbolic() const { return known && symbol; }
  bool isConstant(int64_t value) const {
    return isConstant() && offset == value;
  }
  bool operator==(const DimExpr &other) const {
    return known == other.known && symbol == other.symbol &&
           offset == other.offset;
  }
  bool operator!=(const DimExpr &other) const { return !(*this == other); }
  std::string str() const;
};

using DimShape = llvm::SmallVector<DimExpr, 4>;

/// Reads the integer elements of a constant operation's value attribute.
/// The default reader accepts DenseElementsAttr; the ONNX frontend passes
/// one that also understands its own elements attributes.
using ONNXConstantReader =
    std::function<std::optional<llvm::SmallVector<int64_t>>(mlir::Operation *)>;

/// Symbolic shapes and shape values of one ONNX graph function.
///
/// Every tensor value gets a per-axis extent: a constant, `symbol + offset`
/// for an axis that depends on a bounded dimension, or unknown. Small
/// integer tensors that compute shapes also get their elements as DimExprs.
/// The analysis is recomputed from the IR by every pass that needs it; the
/// graph inputs carry their expressions in `udoncl.dim_exprs` and the module
/// lists the bounds in `udoncl.dimension_bounds`.
class ONNXDimensionExprs {
public:
  explicit ONNXDimensionExprs(mlir::func::FuncOp function,
                              ONNXConstantReader reader = {});

  /// The per-axis extents of `value`, or null when it is not a ranked
  /// tensor whose rank is known.
  const DimShape *getShape(mlir::Value value) const;
  /// The elements of an integer `value` when they are known.
  const DimShape *getIntegers(mlir::Value value) const;
  /// The inclusive bound of a bounded dimension.
  std::optional<int64_t> getBound(mlir::StringAttr symbol) const;
  bool hasBounds() const { return !bounds.empty(); }
  /// The smallest and the largest value of a known expression.
  int64_t minimum(const DimExpr &expression) const;
  int64_t maximum(const DimExpr &expression) const;
  /// The tensor type a shape describes: constant extents are static and
  /// symbolic or unknown extents are dynamic.
  static mlir::RankedTensorType typeOf(const DimShape &shape,
                                       mlir::Type element);
  /// The diagnostics the analysis found: a product of bounded dimensions or
  /// a tensor with more than one bounded dynamic axis.
  llvm::ArrayRef<std::pair<mlir::Operation *, std::string>> getErrors() const {
    return errors;
  }

  // Arithmetic on expressions; unknown when the result is not a constant or
  // `symbol + offset`.
  DimExpr add(const DimExpr &left, const DimExpr &right) const;
  DimExpr subtract(const DimExpr &left, const DimExpr &right) const;
  DimExpr multiply(const DimExpr &left, const DimExpr &right) const;
  DimExpr divide(const DimExpr &left, const DimExpr &right) const;
  /// -1, 0 or 1 when `left` is smaller, equal or greater than `right` for
  /// every extent of the bounded dimensions; std::nullopt otherwise.
  std::optional<int> compare(const DimExpr &left, const DimExpr &right) const;

private:
  void analyze(mlir::Operation *op);
  std::optional<DimShape> inferShape(mlir::Operation *op);
  std::optional<DimShape> inferIntegers(mlir::Operation *op);
  std::optional<DimShape> shapeOf(mlir::Value value) const;
  std::optional<DimShape> integersOf(mlir::Value value) const;
  std::optional<DimShape> broadcast(mlir::Operation *op,
                                    mlir::ValueRange operands);
  std::optional<DimExpr> sliceExtent(const DimExpr &dimension,
                                     const DimExpr &start, const DimExpr &end,
                                     int64_t step);
  void error(mlir::Operation *op, const llvm::Twine &message);

  ONNXConstantReader reader;
  llvm::StringMap<int64_t> bounds;
  llvm::DenseMap<mlir::Value, DimShape> shapes;
  llvm::DenseMap<mlir::Value, DimShape> integers;
  llvm::SmallVector<std::pair<mlir::Operation *, std::string>> errors;
  mlir::Operation *current = nullptr;
};
} // namespace vrc

#endif
