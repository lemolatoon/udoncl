#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "src/Dialect/ONNX/ElementsAttr/DisposableElementsAttr.hpp"

using namespace mlir;

namespace {
DenseElementsAttr constantElements(Value value) {
  Operation *operation = value.getDefiningOp();
  if (!operation)
    return {};
  if (auto dense = operation->getAttrOfType<DenseElementsAttr>("value"))
    return dense;
  if (auto disposable =
          operation->getAttrOfType<DisposableElementsAttr>("value"))
    return disposable.toDenseElementsAttr();
  return {};
}

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
    SmallVector<Operation *> triangular;
    getOperation().walk([&](Operation *op) {
      if (op->getName().getStringRef() == "onnx.Dim")
        dimensions.push_back(op);
      if (op->getName().getStringRef() == "onnx.Trilu")
        triangular.push_back(op);
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
    for (Operation *op : triangular) {
      if (op->getNumOperands() < 1 || op->getNumResults() != 1)
        continue;
      DenseElementsAttr input = constantElements(op->getOperand(0));
      auto type = dyn_cast<RankedTensorType>(op->getResult(0).getType());
      if (!input || !type || !type.hasStaticShape() || type.getRank() < 2 ||
          input.getType() != type)
        continue;
      int64_t diagonal = 0;
      if (op->getNumOperands() >= 2) {
        DenseElementsAttr offset = constantElements(op->getOperand(1));
        if (!offset || offset.getNumElements() != 1 ||
            !isa<IntegerType>(offset.getElementType()))
          continue;
        diagonal = offset.getSplatValue<APInt>().getSExtValue();
      }
      bool upper = true;
      if (auto attribute = op->getAttrOfType<IntegerAttr>("upper"))
        upper = attribute.getInt() != 0;
      int64_t columns = type.getShape().back();
      int64_t rows = type.getShape()[type.getRank() - 2];
      auto retained = [&](int64_t index) {
        int64_t column = index % columns;
        int64_t row = (index / columns) % rows;
        return upper ? column - row >= diagonal : column - row <= diagonal;
      };
      DenseElementsAttr result;
      if (isa<FloatType>(type.getElementType())) {
        SmallVector<APFloat> values;
        values.reserve(type.getNumElements());
        unsigned index = 0;
        APFloat zero = APFloat::getZero(
            cast<FloatType>(type.getElementType()).getFloatSemantics());
        for (APFloat value : input.getValues<APFloat>())
          values.push_back(retained(index++) ? value : zero);
        result = DenseElementsAttr::get(type, values);
      } else if (isa<IntegerType>(type.getElementType())) {
        SmallVector<APInt> values;
        values.reserve(type.getNumElements());
        unsigned width = type.getElementTypeBitWidth();
        unsigned index = 0;
        for (APInt value : input.getValues<APInt>())
          values.push_back(retained(index++) ? value : APInt(width, 0));
        result = DenseElementsAttr::get(type, values);
      } else {
        continue;
      }
      OpBuilder builder(op);
      Value constant = builder.create<arith::ConstantOp>(op->getLoc(), result);
      op->getResult(0).replaceAllUsesWith(constant);
      op->erase();
    }
  }
};
static PassRegistration<MaterializeONNXConstantsPass> registration;
} // namespace
