#include "VRC/Conversion/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_SIMPLIFYONNXEMPTYCONCATPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {
bool isStaticallyEmpty(Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  return type && type.hasStaticShape() && type.getNumElements() == 0;
}

/// Removes statically empty operands from an ONNX Concat. A Concat left with
/// one operand of the result type is replaced by that operand.
struct DropEmptyConcatOperands : RewritePattern {
  explicit DropEmptyConcatOperands(MLIRContext *context)
      : RewritePattern("onnx.Concat", /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *concat,
                                PatternRewriter &rewriter) const override {
    if (concat->getNumResults() != 1)
      return rewriter.notifyMatchFailure(concat, "expected one result");
    auto resultType =
        dyn_cast<RankedTensorType>(concat->getResult(0).getType());
    if (!resultType || !resultType.hasStaticShape() ||
        resultType.getNumElements() == 0)
      return rewriter.notifyMatchFailure(concat, "result is not non-empty");
    SmallVector<Value> operands;
    for (Value operand : concat->getOperands())
      if (!isStaticallyEmpty(operand))
        operands.push_back(operand);
    if (operands.size() == concat->getNumOperands() || operands.empty())
      return rewriter.notifyMatchFailure(concat, "no removable empty operand");
    if (operands.size() == 1) {
      if (operands.front().getType() != resultType)
        return rewriter.notifyMatchFailure(concat,
                                           "remaining operand type differs");
      rewriter.replaceOp(concat, operands.front());
      return success();
    }
    rewriter.modifyOpInPlace(concat, [&] { concat->setOperands(operands); });
    return success();
  }
};

struct SimplifyONNXEmptyConcatPass
    : impl::SimplifyONNXEmptyConcatPassBase<SimplifyONNXEmptyConcatPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<DropEmptyConcatOperands>(&getContext());
    GreedyRewriteConfig config;
    config.enableFolding(false);
    config.enableConstantCSE(false);
    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns), config)))
      signalPassFailure();
  }
};
} // namespace
} // namespace vrc
