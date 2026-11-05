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
    return "Convert ONNX disposable initializer attributes to builtin dense "
           "attributes";
  }
  void runOnOperation() override {
    getOperation().walk([](Operation *op) {
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
  }
};
static PassRegistration<MaterializeONNXConstantsPass> registration;
} // namespace
