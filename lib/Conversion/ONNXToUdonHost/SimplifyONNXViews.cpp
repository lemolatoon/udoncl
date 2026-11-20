#include "VRC/Conversion/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Dialect/ONNX/ElementsAttr/DisposableElementsAttr.hpp"

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_SIMPLIFYONNXVIEWSPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {
RankedTensorType staticNonEmptyType(Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || !type.hasStaticShape() || type.getNumElements() == 0)
    return {};
  return type;
}

DenseElementsAttr constantElements(Value value) {
  Operation *operation = value.getDefiningOp();
  if (!operation)
    return {};
  StringRef name = operation->getName().getStringRef();
  if (name != "onnx.Constant" && name != "arith.constant")
    return {};
  if (auto dense = operation->getAttrOfType<DenseElementsAttr>("value"))
    return dense;
  if (auto disposable =
          operation->getAttrOfType<DisposableElementsAttr>("value"))
    return disposable.toDenseElementsAttr();
  return {};
}

bool isAbsentOperand(Value value) { return isa<NoneType>(value.getType()); }

/// Shares one shape constant per function and shape. The greedy driver may
/// erase a constant whose Reshape became dead, so erased operations leave the
/// cache.
struct ShapeConstantCache : RewriterBase::Listener {
  using Key = std::pair<Operation *, Attribute>;
  llvm::DenseMap<Key, Operation *> constants;

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
      Block &entry = function->getRegion(0).front();
      rewriter.setInsertionPointToStart(&entry);
      cached = rewriter.create<arith::ConstantOp>(user->getLoc(), value)
                   .getOperation();
    }
    return cached->getResult(0);
  }
};

/// A Transpose that keeps the relative order of every non-unit dimension
/// leaves the row-major element order unchanged, so it is a Reshape.
struct TransposeOfUnitDimensionsToReshape : RewritePattern {
  TransposeOfUnitDimensionsToReshape(MLIRContext *context,
                                     ShapeConstantCache &shapes)
      : RewritePattern("onnx.Transpose", /*benefit=*/1, context),
        shapes(shapes) {}

  LogicalResult matchAndRewrite(Operation *transpose,
                                PatternRewriter &rewriter) const override {
    if (transpose->getNumOperands() != 1 || transpose->getNumResults() != 1)
      return rewriter.notifyMatchFailure(transpose, "unexpected signature");
    RankedTensorType input = staticNonEmptyType(transpose->getOperand(0));
    RankedTensorType output = staticNonEmptyType(transpose->getResult(0));
    if (!input || !output || input.getRank() != output.getRank() ||
        input.getElementType() != output.getElementType())
      return rewriter.notifyMatchFailure(transpose, "shapes are not static");
    int64_t rank = input.getRank();
    SmallVector<int64_t> permutation;
    if (auto dense = transpose->getAttrOfType<DenseI64ArrayAttr>("perm")) {
      permutation.assign(dense.asArrayRef().begin(), dense.asArrayRef().end());
    } else if (auto array = transpose->getAttrOfType<ArrayAttr>("perm")) {
      for (Attribute element : array) {
        auto integer = dyn_cast<IntegerAttr>(element);
        if (!integer)
          return rewriter.notifyMatchFailure(transpose, "malformed perm");
        permutation.push_back(integer.getInt());
      }
    } else if (!transpose->getAttr("perm")) {
      for (int64_t axis = rank; axis-- > 0;)
        permutation.push_back(axis);
    }
    if (permutation.size() != static_cast<size_t>(rank))
      return rewriter.notifyMatchFailure(transpose, "malformed perm");
    SmallVector<bool> seen(rank, false);
    int64_t previousNonUnitAxis = -1;
    for (auto [outputAxis, inputAxis] : llvm::enumerate(permutation)) {
      if (inputAxis < 0 || inputAxis >= rank || seen[inputAxis] ||
          output.getDimSize(outputAxis) != input.getDimSize(inputAxis))
        return rewriter.notifyMatchFailure(transpose, "malformed perm");
      seen[inputAxis] = true;
      if (input.getDimSize(inputAxis) == 1)
        continue;
      if (inputAxis < previousNonUnitAxis)
        return rewriter.notifyMatchFailure(transpose,
                                           "reorders non-unit dimensions");
      previousNonUnitAxis = inputAxis;
    }
    Value shape = shapes.get(rewriter, transpose, output.getShape());
    OperationState state(transpose->getLoc(), "onnx.Reshape");
    state.addOperands({transpose->getOperand(0), shape});
    state.addTypes(output);
    if (Attribute name = transpose->getAttr("onnx_node_name"))
      state.addAttribute("onnx_node_name", name);
    Operation *reshape = rewriter.create(state);
    rewriter.replaceOp(transpose, reshape->getResults());
    return success();
  }

  ShapeConstantCache &shapes;
};

/// A unit-step Slice whose static result shape equals its input shape selects
/// every element in order.
struct RemoveIdentitySlice : RewritePattern {
  explicit RemoveIdentitySlice(MLIRContext *context)
      : RewritePattern("onnx.Slice", /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *slice,
                                PatternRewriter &rewriter) const override {
    if (slice->getNumOperands() < 3 || slice->getNumResults() != 1)
      return rewriter.notifyMatchFailure(slice, "unexpected signature");
    RankedTensorType input = staticNonEmptyType(slice->getOperand(0));
    if (!input || slice->getResult(0).getType() != input)
      return rewriter.notifyMatchFailure(slice, "result differs from input");
    if (slice->getNumOperands() >= 5 &&
        !isAbsentOperand(slice->getOperand(4))) {
      DenseElementsAttr steps = constantElements(slice->getOperand(4));
      if (!steps || !isa<IntegerType>(steps.getElementType()) ||
          llvm::any_of(steps.getValues<APInt>(),
                       [](const APInt &step) { return !step.isOne(); }))
        return rewriter.notifyMatchFailure(slice, "steps are not all one");
    }
    rewriter.replaceOp(slice, slice->getOperand(0));
    return success();
  }
};

/// An Expand whose static result shape equals its input shape broadcasts
/// nothing.
struct RemoveIdentityExpand : RewritePattern {
  explicit RemoveIdentityExpand(MLIRContext *context)
      : RewritePattern("onnx.Expand", /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *expand,
                                PatternRewriter &rewriter) const override {
    if (expand->getNumOperands() != 2 || expand->getNumResults() != 1)
      return rewriter.notifyMatchFailure(expand, "unexpected signature");
    RankedTensorType input = staticNonEmptyType(expand->getOperand(0));
    if (!input || expand->getResult(0).getType() != input)
      return rewriter.notifyMatchFailure(expand, "result differs from input");
    rewriter.replaceOp(expand, expand->getOperand(0));
    return success();
  }
};

struct SimplifyONNXViewsPass
    : impl::SimplifyONNXViewsPassBase<SimplifyONNXViewsPass> {
  void runOnOperation() override {
    ShapeConstantCache shapes;
    RewritePatternSet patterns(&getContext());
    patterns.add<TransposeOfUnitDimensionsToReshape>(&getContext(), shapes);
    patterns.add<RemoveIdentitySlice>(&getContext());
    patterns.add<RemoveIdentityExpand>(&getContext());
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
