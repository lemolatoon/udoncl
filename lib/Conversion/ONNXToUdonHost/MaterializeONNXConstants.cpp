#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "src/Dialect/ONNX/ElementsAttr/DisposableElementsAttr.hpp"

using namespace mlir;

namespace {
struct MaterializeONNXConstantsPass
    : PassWrapper<MaterializeONNXConstantsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializeONNXConstantsPass)
  StringRef getArgument() const final {
    return "udoncl-materialize-onnx-constants";
  }
  StringRef getDescription() const final {
    return "Materialize ONNX constants and statically known shape values";
  }
  void runOnOperation() override {
    SmallVector<Operation *> dimensions;
    getOperation().walk([&](Operation *op) {
      if (op->getName().getStringRef() == "onnx.Dim")
        dimensions.push_back(op);
      if (op->getName().getStringRef() != "onnx.Constant")
        return;
      auto value = op->getAttrOfType<DisposableElementsAttr>("value");
      if (!value)
        return;
      DenseElementsAttr dense = value.toDenseElementsAttr();
      auto tensor = cast<RankedTensorType>(dense.getType());
      auto integer = dyn_cast<IntegerType>(tensor.getElementType());
      if (integer && integer.isUnsigned()) {
        auto element = IntegerType::get(op->getContext(), integer.getWidth());
        auto type = RankedTensorType::get(tensor.getShape(), element,
                                          tensor.getEncoding());
        dense = DenseElementsAttr::getFromRawBuffer(type, dense.getRawData());
        op->getResult(0).setType(type);
      }
      op->setAttr("value", dense);
    });
    for (Operation *op : dimensions) {
      if (op->getNumOperands() != 1 || op->getNumResults() != 1)
        continue;
      auto input = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
      auto result = dyn_cast<RankedTensorType>(op->getResult(0).getType());
      auto axis = op->getAttrOfType<IntegerAttr>("axis");
      if (!input || !result || !result.hasStaticShape() || !axis)
        continue;
      int64_t index = axis.getInt();
      if (index < 0)
        index += input.getRank();
      if (index < 0 || index >= input.getRank() || input.isDynamicDim(index))
        continue;
      auto value =
          DenseElementsAttr::get(result, APInt(result.getElementTypeBitWidth(),
                                               input.getDimSize(index), true));
      OpBuilder builder(op);
      Value constant = builder.create<arith::ConstantOp>(op->getLoc(), value);
      op->getResult(0).replaceAllUsesWith(constant);
      op->erase();
    }
  }
};
static PassRegistration<MaterializeONNXConstantsPass> registration;
} // namespace
