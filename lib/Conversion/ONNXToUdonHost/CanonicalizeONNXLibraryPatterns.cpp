#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Dialect/ONNX/ElementsAttr/DisposableElementsAttr.hpp"

using namespace mlir;

namespace {
bool named(Operation *operation, StringRef name) {
  return operation && operation->getName().getStringRef() == name;
}

Value otherOperand(Operation *operation, Value value) {
  if (!operation || operation->getNumOperands() != 2)
    return {};
  if (operation->getOperand(0) == value)
    return operation->getOperand(1);
  if (operation->getOperand(1) == value)
    return operation->getOperand(0);
  return {};
}

DenseElementsAttr denseConstant(Value value) {
  Operation *operation = value.getDefiningOp();
  if (!named(operation, "onnx.Constant"))
    return {};
  if (auto dense = operation->getAttrOfType<DenseElementsAttr>("value"))
    return dense;
  if (auto disposable =
          operation->getAttrOfType<DisposableElementsAttr>("value"))
    return disposable.toDenseElementsAttr();
  return {};
}

bool isScalarFloatConstant(Value value, double expected) {
  DenseElementsAttr dense = denseConstant(value);
  if (!dense || dense.getNumElements() != 1 ||
      !isa<FloatType>(dense.getElementType()))
    return false;
  return dense.getSplatValue<APFloat>().convertToDouble() == expected;
}

bool hasLastAxisReduction(Operation *operation) {
  if (!named(operation, "onnx.ReduceMeanV13"))
    return false;
  bool keepDimensions = true;
  if (auto keep = operation->getAttrOfType<IntegerAttr>("keepdims"))
    keepDimensions = keep.getInt() != 0;
  if (!keepDimensions)
    return false;
  if (auto axes = operation->getAttrOfType<DenseI64ArrayAttr>("axes"))
    return axes.size() == 1 && axes[0] == -1;
  if (auto axes = operation->getAttrOfType<ArrayAttr>("axes"))
    return axes.size() == 1 && cast<IntegerAttr>(axes[0]).getInt() == -1;
  return false;
}

bool isSquareOf(Operation *operation, Value input) {
  if (named(operation, "onnx.Mul"))
    return operation->getNumOperands() == 2 &&
           operation->getOperand(0) == input &&
           operation->getOperand(1) == input;
  return named(operation, "onnx.Pow") && operation->getNumOperands() == 2 &&
         operation->getOperand(0) == input &&
         isScalarFloatConstant(operation->getOperand(1), 2.0);
}

struct RMSNormMatch {
  Value input;
  Value weight;
  Value epsilon;
};

std::optional<RMSNormMatch> matchRMSNorm(Operation *root) {
  if (!named(root, "onnx.Mul") || root->getNumResults() != 1)
    return std::nullopt;
  for (unsigned normalizedIndex = 0; normalizedIndex != 2; ++normalizedIndex) {
    Value normalized = root->getOperand(normalizedIndex);
    Value weight = root->getOperand(1 - normalizedIndex);
    Operation *normalize = normalized.getDefiningOp();
    auto weightType = dyn_cast<RankedTensorType>(weight.getType());
    if (!named(normalize, "onnx.Mul") || !weightType ||
        !weightType.hasStaticShape() || weightType.getRank() == 0)
      continue;
    for (unsigned inputIndex = 0; inputIndex != 2; ++inputIndex) {
      Value input = normalize->getOperand(inputIndex);
      Value reciprocal = normalize->getOperand(1 - inputIndex);
      Operation *divide = reciprocal.getDefiningOp();
      if (!named(divide, "onnx.Div") || divide->getNumOperands() != 2 ||
          !isScalarFloatConstant(divide->getOperand(0), 1.0))
        continue;
      Operation *squareRoot = divide->getOperand(1).getDefiningOp();
      if (!named(squareRoot, "onnx.Sqrt") || squareRoot->getNumOperands() != 1)
        continue;
      Operation *add = squareRoot->getOperand(0).getDefiningOp();
      if (!named(add, "onnx.Add"))
        continue;
      Operation *mean = nullptr;
      Value epsilon;
      for (unsigned meanIndex = 0; meanIndex != 2; ++meanIndex) {
        Operation *candidate = add->getOperand(meanIndex).getDefiningOp();
        Value candidateEpsilon = add->getOperand(1 - meanIndex);
        DenseElementsAttr epsilonConstant = denseConstant(candidateEpsilon);
        if (hasLastAxisReduction(candidate) && epsilonConstant &&
            epsilonConstant.getNumElements() == 1 &&
            isa<FloatType>(epsilonConstant.getElementType())) {
          mean = candidate;
          epsilon = candidateEpsilon;
          break;
        }
      }
      if (!mean || mean->getNumOperands() != 1)
        continue;
      Operation *square = mean->getOperand(0).getDefiningOp();
      if (!isSquareOf(square, input))
        continue;
      return RMSNormMatch{input, weight, epsilon};
    }
  }
  return std::nullopt;
}

/// Replaces a generic ONNX RMS normalization subgraph rooted at its final
/// weight multiplication with the public RmsNorm library operation.
struct RMSNormPattern : RewritePattern {
  explicit RMSNormPattern(MLIRContext *context)
      : RewritePattern("onnx.Mul", /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *root,
                                PatternRewriter &rewriter) const override {
    auto match = matchRMSNorm(root);
    if (!match)
      return rewriter.notifyMatchFailure(root, "not an RMS normalization");
    OperationState state(root->getLoc(), "onnx.Custom");
    state.addOperands({match->input, match->weight, match->epsilon});
    state.addTypes(root->getResultTypes());
    state.addAttribute("function_name", rewriter.getStringAttr("RmsNorm"));
    state.addAttribute("domain_name",
                       rewriter.getStringAttr("com.lemolatoon.udoncl"));
    Operation *replacement = rewriter.create(state);
    rewriter.replaceOp(root, replacement->getResults());
    return success();
  }
};

struct CanonicalizeONNXLibraryPatternsPass
    : PassWrapper<CanonicalizeONNXLibraryPatternsPass,
                  OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      CanonicalizeONNXLibraryPatternsPass)
  StringRef getArgument() const final {
    return "udoncl-canonicalize-onnx-library-patterns";
  }
  StringRef getDescription() const final {
    return "Canonicalize generic ONNX subgraphs to public Udon library ops";
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<RMSNormPattern>(&getContext());
    // The replaced subgraph becomes trivially dead and is erased by the
    // driver. Folding stays disabled so this pass only performs the library
    // pattern rewrites.
    GreedyRewriteConfig config;
    config.enableFolding(false);
    config.enableConstantCSE(false);
    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns), config)))
      signalPassFailure();
  }
};

static PassRegistration<CanonicalizeONNXLibraryPatternsPass> registration;
} // namespace
