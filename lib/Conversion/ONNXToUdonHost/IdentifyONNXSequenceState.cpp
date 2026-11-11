#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;

namespace {
struct IdentifyONNXSequenceStatePass
    : PassWrapper<IdentifyONNXSequenceStatePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IdentifyONNXSequenceStatePass)

  StringRef getArgument() const final {
    return "udoncl-identify-onnx-sequence-state";
  }
  StringRef getDescription() const final {
    return "Identify append-only ONNX graph inputs and outputs that can use "
           "fixed-capacity sequence state banks";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = module.getContext();
    Builder builder(context);
    SmallVector<Attribute> states;

    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      if (function.isDeclaration() || !function.getBody().hasOneBlock())
        continue;
      Operation *returnOp = function.getBody().front().getTerminator();
      if (!returnOp || (returnOp->getName().getStringRef() != "onnx.Return" &&
                        returnOp->getName().getStringRef() != "func.return"))
        continue;

      for (auto [outputIndex, output] :
           llvm::enumerate(returnOp->getOperands())) {
        Operation *concat = output.getDefiningOp();
        if (!concat || concat->getName().getStringRef() != "onnx.Concat" ||
            concat->getNumResults() != 1 || concat->getNumOperands() < 2)
          continue;

        auto input = dyn_cast<BlockArgument>(concat->getOperand(0));
        if (!input || input.getOwner() != &function.getBody().front())
          continue;
        auto inputType = dyn_cast<RankedTensorType>(input.getType());
        auto outputType = dyn_cast<RankedTensorType>(output.getType());
        auto axisAttr = concat->getAttrOfType<IntegerAttr>("axis");
        if (!inputType || !outputType || !inputType.hasStaticShape() ||
            !outputType.hasStaticShape() || !axisAttr ||
            inputType.getRank() != outputType.getRank() ||
            inputType.getElementType() != outputType.getElementType())
          continue;

        int64_t axis = axisAttr.getInt();
        if (axis < 0)
          axis += inputType.getRank();
        if (axis < 0 || axis >= inputType.getRank())
          continue;
        bool compatible = true;
        for (int64_t dimension = 0; dimension < inputType.getRank();
             ++dimension)
          if (dimension != axis && inputType.getDimSize(dimension) !=
                                       outputType.getDimSize(dimension))
            compatible = false;
        int64_t capacity = outputType.getDimSize(axis);
        int64_t step = capacity - inputType.getDimSize(axis);
        if (!compatible || capacity <= 0 || step <= 0)
          continue;

        states.push_back(builder.getDictionaryAttr({
            builder.getNamedAttr(
                "input_index", builder.getI64IntegerAttr(input.getArgNumber())),
            builder.getNamedAttr("output_index",
                                 builder.getI64IntegerAttr(outputIndex)),
            builder.getNamedAttr("axis", builder.getI64IntegerAttr(axis)),
            builder.getNamedAttr("step", builder.getI64IntegerAttr(step)),
            builder.getNamedAttr("capacity",
                                 builder.getI64IntegerAttr(capacity)),
        }));
      }
    }
    module->setAttr("udoncl.onnx_sequence_states",
                    builder.getArrayAttr(states));
  }
};

static PassRegistration<IdentifyONNXSequenceStatePass> registration;
} // namespace
