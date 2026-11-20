// Folds the repetition of a MatMul right operand into the rows of the left
// operand.
//
// Grouped-query attention repeats every key/value head for a group of query
// heads: Expand broadcasts a unit axis of X[P..., G, 1, K, N] to R and a
// Reshape merges it into the batch, B[P..., G*R, K, N]. A MatMul with that B
// multiplies R consecutive left matrices by the same right matrix, which is
// one MatMul whose left matrices are those R matrices stacked as rows:
//
//   MatMul(A[P..., G*R, M, K], B)
//     = Reshape(MatMul(Reshape(A)[P..., G, R*M, K], Reshape(X)[P..., G, K, N]))
//
// Every output element is the same dot product of the same operands, so the
// result is unchanged while the repeated operand is never materialized.

#include "VRC/Conversion/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Dialect/ONNX/ElementsAttr/DisposableElementsAttr.hpp"

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_FOLDONNXREPEATEDMATMULOPERANDPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {
RankedTensorType staticType(Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || !type.hasStaticShape() || type.getNumElements() == 0)
    return {};
  return type;
}

Operation *producer(Value value, StringRef name) {
  Operation *operation = value.getDefiningOp();
  if (!operation || operation->getName().getStringRef() != name ||
      operation->getNumResults() != 1)
    return nullptr;
  return operation;
}

/// Shares one shape constant per function and shape. The greedy driver may
/// erase a constant that became dead, so erased operations leave the cache.
struct ShapeConstantCache : RewriterBase::Listener {
  llvm::DenseMap<std::pair<Operation *, Attribute>, Operation *> constants;

  void notifyOperationErased(Operation *operation) override {
    for (auto iterator = constants.begin(); iterator != constants.end();
         ++iterator)
      if (iterator->second == operation) {
        constants.erase(iterator);
        return;
      }
  }

  Value get(PatternRewriter &rewriter, Operation *user,
            ArrayRef<int64_t> shape) {
    Operation *function = user->getParentOp();
    while (function && !function->hasTrait<OpTrait::IsIsolatedFromAbove>())
      function = function->getParentOp();
    auto type = RankedTensorType::get({static_cast<int64_t>(shape.size())},
                                      rewriter.getI64Type());
    DenseElementsAttr value = DenseIntElementsAttr::get(type, shape);
    Operation *&cached = constants[{function, value}];
    if (!cached) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&function->getRegion(0).front());
      cached = arith::ConstantOp::create(rewriter, user->getLoc(), value)
                   .getOperation();
    }
    return cached->getResult(0);
  }
};

struct FoldRepeatedRightOperand : RewritePattern {
  FoldRepeatedRightOperand(MLIRContext *context, ShapeConstantCache &shapes)
      : RewritePattern("onnx.MatMul", /*benefit=*/1, context), shapes(shapes) {}

  ShapeConstantCache &shapes;

  LogicalResult matchAndRewrite(Operation *matmul,
                                PatternRewriter &rewriter) const override {
    if (matmul->getNumOperands() != 2 || matmul->getNumResults() != 1)
      return rewriter.notifyMatchFailure(matmul, "unexpected signature");
    Value left = matmul->getOperand(0);
    Operation *reshape = producer(matmul->getOperand(1), "onnx.Reshape");
    Operation *expand =
        reshape ? producer(reshape->getOperand(0), "onnx.Expand") : nullptr;
    if (!expand)
      return rewriter.notifyMatchFailure(matmul, "right operand not repeated");
    Value source = expand->getOperand(0);
    RankedTensorType a = staticType(left);
    RankedTensorType b = staticType(matmul->getOperand(1));
    RankedTensorType c = staticType(matmul->getResult(0));
    RankedTensorType expanded = staticType(expand->getResult(0));
    RankedTensorType x = staticType(source);
    if (!a || !b || !c || !expanded || !x || b.getRank() < 3 ||
        a.getRank() != b.getRank() || c.getRank() != b.getRank() ||
        x.getRank() != b.getRank() + 1 || expanded.getRank() != x.getRank() ||
        a.getElementType() != b.getElementType() ||
        x.getElementType() != b.getElementType())
      return rewriter.notifyMatchFailure(matmul, "shapes are not static");

    // B[P..., H, K, N] with H = G * R, the Expand result [P..., G, R, K, N]
    // and its source [P..., G, 1, K, N]: only the repeat axis broadcasts.
    int64_t rank = b.getRank();
    int64_t batch = rank - 3;
    ArrayRef<int64_t> bShape = b.getShape();
    ArrayRef<int64_t> eShape = expanded.getShape();
    ArrayRef<int64_t> xShape = x.getShape();
    int64_t groups = eShape[batch];
    int64_t repeats = eShape[batch + 1];
    int64_t m = a.getShape()[rank - 2];
    if (repeats < 2 || xShape[batch + 1] != 1 ||
        groups * repeats != bShape[batch] ||
        eShape.take_front(batch) != bShape.take_front(batch) ||
        eShape.take_back(2) != bShape.take_back(2))
      return rewriter.notifyMatchFailure(matmul, "not a repeated batch");
    for (int64_t axis = 0; axis < x.getRank(); ++axis)
      if (axis != batch + 1 && xShape[axis] != eShape[axis])
        return rewriter.notifyMatchFailure(matmul, "source broadcasts more");
    // The left operand has the full batch, so its matrices regroup as rows.
    if (a.getShape().drop_back(2) != bShape.drop_back(2) ||
        a.getShape().back() != bShape[rank - 2] ||
        c.getShape().drop_back(2) != bShape.drop_back(2) ||
        c.getShape()[rank - 2] != m || c.getShape().back() != bShape.back())
      return rewriter.notifyMatchFailure(matmul, "left operand broadcasts");

    Location location = matmul->getLoc();
    auto reshaped = [&](Value value, ArrayRef<int64_t> shape) -> Value {
      auto type = RankedTensorType::get(
          shape, cast<RankedTensorType>(value.getType()).getElementType());
      OperationState state(location, "onnx.Reshape");
      state.addOperands({value, shapes.get(rewriter, matmul, shape)});
      state.addTypes(type);
      return rewriter.create(state)->getResult(0);
    };
    SmallVector<int64_t> rowsShape(bShape.take_front(batch));
    rowsShape.append({groups, repeats * m, a.getShape().back()});
    SmallVector<int64_t> matrixShape(bShape.take_front(batch));
    matrixShape.append({groups, bShape[rank - 2], bShape.back()});
    SmallVector<int64_t> productShape(bShape.take_front(batch));
    productShape.append({groups, repeats * m, bShape.back()});

    Value rows = reshaped(left, rowsShape);
    Value matrices = reshaped(source, matrixShape);
    OperationState state(location, "onnx.MatMul");
    state.addOperands({rows, matrices});
    state.addTypes(RankedTensorType::get(productShape, c.getElementType()));
    for (NamedAttribute attribute : matmul->getDiscardableAttrs())
      state.addAttribute(attribute.getName(), attribute.getValue());
    Operation *product = rewriter.create(state);
    rewriter.replaceOp(matmul, reshaped(product->getResult(0), c.getShape()));
    return success();
  }
};

/// The key operand of attention transposes its last two axes and may scale
/// every element by a constant scalar after the repetition:
///
///   MatMul(A, Mul(Transpose(Reshape(Expand(X))), s))
///
/// Both commute with the repetition element for element: every element of
/// the result is s * the same element of X. Moving them in front of the
/// Expand,
///
///   MatMul(A, Reshape(Expand(Mul(Transpose(X), s))))
///
/// computes every product once per element of X instead of once per repeat
/// and leaves a right operand that FoldRepeatedRightOperand folds.
struct SinkRepeatBelowTransposeAndScale : RewritePattern {
  SinkRepeatBelowTransposeAndScale(MLIRContext *context,
                                   ShapeConstantCache &shapes)
      : RewritePattern("onnx.MatMul", /*benefit=*/2, context), shapes(shapes) {}

  ShapeConstantCache &shapes;

  static std::optional<float> scalarConstant(Value value) {
    Operation *operation = value.getDefiningOp();
    if (!operation)
      return std::nullopt;
    StringRef name = operation->getName().getStringRef();
    if (name != "onnx.Constant" && name != "arith.constant")
      return std::nullopt;
    DenseElementsAttr dense =
        operation->getAttrOfType<DenseElementsAttr>("value");
    if (auto disposable =
            operation->getAttrOfType<DisposableElementsAttr>("value"))
      dense = disposable.toDenseElementsAttr();
    if (!dense || dense.getNumElements() != 1 ||
        !dense.getElementType().isF32())
      return std::nullopt;
    return (*dense.getValues<APFloat>().begin()).convertToFloat();
  }

  static bool swapsLastTwoAxes(Operation *transpose, int64_t rank) {
    SmallVector<int64_t> permutation;
    if (auto dense = transpose->getAttrOfType<DenseI64ArrayAttr>("perm")) {
      permutation.assign(dense.asArrayRef().begin(), dense.asArrayRef().end());
    } else if (auto array = transpose->getAttrOfType<ArrayAttr>("perm")) {
      for (Attribute element : array) {
        auto integer = dyn_cast<IntegerAttr>(element);
        if (!integer)
          return false;
        permutation.push_back(integer.getInt());
      }
    }
    if (static_cast<int64_t>(permutation.size()) != rank || rank < 2)
      return false;
    for (int64_t axis = 0; axis < rank - 2; ++axis)
      if (permutation[axis] != axis)
        return false;
    return permutation[rank - 2] == rank - 1 &&
           permutation[rank - 1] == rank - 2;
  }

  LogicalResult matchAndRewrite(Operation *matmul,
                                PatternRewriter &rewriter) const override {
    if (matmul->getNumOperands() != 2 || matmul->getNumResults() != 1)
      return rewriter.notifyMatchFailure(matmul, "unexpected signature");
    Value right = matmul->getOperand(1);
    Operation *scale = producer(right, "onnx.Mul");
    Value scaleFactor;
    Value transposed = right;
    if (scale) {
      if (scale->getNumOperands() != 2 || !scale->getResult(0).hasOneUse() ||
          !scalarConstant(scale->getOperand(1)) ||
          scale->getResult(0).getType() != scale->getOperand(0).getType())
        return rewriter.notifyMatchFailure(matmul, "not a scalar scale");
      scaleFactor = scale->getOperand(1);
      transposed = scale->getOperand(0);
    }
    Operation *transpose = producer(transposed, "onnx.Transpose");
    Operation *reshape =
        transpose ? producer(transpose->getOperand(0), "onnx.Reshape")
                  : nullptr;
    Operation *expand =
        reshape ? producer(reshape->getOperand(0), "onnx.Expand") : nullptr;
    if (!expand || !transpose->getResult(0).hasOneUse() ||
        !reshape->getResult(0).hasOneUse() || !expand->getResult(0).hasOneUse())
      return rewriter.notifyMatchFailure(matmul, "no repeat below a view");
    RankedTensorType b = staticType(transposed);
    RankedTensorType merged = staticType(reshape->getResult(0));
    RankedTensorType expanded = staticType(expand->getResult(0));
    RankedTensorType x = staticType(expand->getOperand(0));
    if (!b || !merged || !expanded || !x || merged.getRank() < 3 ||
        x.getRank() != merged.getRank() + 1 ||
        expanded.getRank() != x.getRank() ||
        !swapsLastTwoAxes(transpose, merged.getRank()))
      return rewriter.notifyMatchFailure(matmul, "shapes are not static");
    // The repetition must be the one FoldRepeatedRightOperand folds: only
    // the axis after the groups broadcasts, and the Reshape merges it.
    int64_t rank = merged.getRank();
    int64_t batch = rank - 3;
    ArrayRef<int64_t> eShape = expanded.getShape();
    ArrayRef<int64_t> xShape = x.getShape();
    ArrayRef<int64_t> mShape = merged.getShape();
    if (xShape[batch + 1] != 1 || eShape[batch + 1] < 2 ||
        eShape[batch] * eShape[batch + 1] != mShape[batch] ||
        eShape.take_front(batch) != mShape.take_front(batch) ||
        eShape.take_back(2) != mShape.take_back(2))
      return rewriter.notifyMatchFailure(matmul, "not a repeated batch");
    for (int64_t axis = 0; axis < x.getRank(); ++axis)
      if (axis != batch + 1 && xShape[axis] != eShape[axis])
        return rewriter.notifyMatchFailure(matmul, "source broadcasts more");

    Location location = matmul->getLoc();
    Type element = x.getElementType();
    int64_t xRank = x.getRank();
    SmallVector<int64_t> permutation;
    for (int64_t axis = 0; axis < xRank - 2; ++axis)
      permutation.push_back(axis);
    permutation.append({xRank - 1, xRank - 2});
    SmallVector<int64_t> turnedShape(xShape);
    std::swap(turnedShape[xRank - 1], turnedShape[xRank - 2]);
    OperationState turnState(location, "onnx.Transpose");
    turnState.addOperands(expand->getOperand(0));
    turnState.addAttribute("perm", rewriter.getI64ArrayAttr(permutation));
    turnState.addTypes(RankedTensorType::get(turnedShape, element));
    Value turned = rewriter.create(turnState)->getResult(0);
    if (scaleFactor) {
      OperationState scaleState(location, "onnx.Mul");
      scaleState.addOperands({turned, scaleFactor});
      scaleState.addTypes(turned.getType());
      turned = rewriter.create(scaleState)->getResult(0);
    }
    SmallVector<int64_t> repeatedShape(eShape);
    std::swap(repeatedShape[xRank - 1], repeatedShape[xRank - 2]);
    OperationState expandState(location, "onnx.Expand");
    expandState.addOperands(
        {turned, shapes.get(rewriter, matmul, repeatedShape)});
    expandState.addTypes(RankedTensorType::get(repeatedShape, element));
    Value repeated = rewriter.create(expandState)->getResult(0);
    OperationState mergeState(location, "onnx.Reshape");
    mergeState.addOperands(
        {repeated, shapes.get(rewriter, matmul, b.getShape())});
    mergeState.addTypes(b);
    Value operand = rewriter.create(mergeState)->getResult(0);
    rewriter.modifyOpInPlace(matmul, [&] { matmul->setOperand(1, operand); });
    return success();
  }
};

struct FoldONNXRepeatedMatMulOperandPass
    : impl::FoldONNXRepeatedMatMulOperandPassBase<
          FoldONNXRepeatedMatMulOperandPass> {
  void runOnOperation() override {
    ShapeConstantCache shapes;
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldRepeatedRightOperand>(&getContext(), shapes);
    patterns.add<SinkRepeatBelowTransposeAndScale>(&getContext(), shapes);
    GreedyRewriteConfig config;
    config.enableFolding(false);
    config.enableConstantCSE(false);
    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);
    config.setListener(&shapes);
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns), config)))
      signalPassFailure();
  }
};
} // namespace
} // namespace vrc
