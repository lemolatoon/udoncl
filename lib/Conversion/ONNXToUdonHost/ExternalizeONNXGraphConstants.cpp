#include "ONNXStaticTensor.h"
#include "VRC/Conversion/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/FileSystem.h"

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_EXTERNALIZEONNXGRAPHCONSTANTSPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {
/// Returns the dense value of a graph-local tensor constant, or null when
/// `operation` is not one. Initializers were already replaced by static
/// tensor calls, so every remaining `onnx.Constant` is graph-local.
DenseElementsAttr graphConstantValue(Operation *operation) {
  if (auto constant = dyn_cast<arith::ConstantOp>(operation))
    return dyn_cast<DenseElementsAttr>(constant.getValue());
  if (operation->getName().getStringRef() == "onnx.Constant" &&
      operation->getNumResults() == 1)
    return operation->getAttrOfType<DenseElementsAttr>("value");
  return {};
}

struct ExternalizeONNXGraphConstantsPass
    : impl::ExternalizeONNXGraphConstantsPassBase<
          ExternalizeONNXGraphConstantsPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto existing = module->getAttrOfType<ArrayAttr>(onnxStaticTensorsAttr);
    if (!existing) {
      module.emitError("graph constant externalization requires ONNX "
                       "initializer externalization");
      return signalPassFailure();
    }
    int64_t nextId = 0;
    for (Attribute attribute : existing) {
      auto record = dyn_cast<DictionaryAttr>(attribute);
      auto id = record ? record.getAs<IntegerAttr>("id") : IntegerAttr();
      if (!id) {
        module.emitError("static tensor record has no id");
        return signalPassFailure();
      }
      nextId = std::max<int64_t>(nextId, id.getInt() + 1);
    }

    SmallVector<std::pair<Operation *, DenseElementsAttr>> constants;
    module.walk([&](Operation *operation) {
      DenseElementsAttr value = graphConstantValue(operation);
      if (isStaticTensorRepresentable(value) &&
          value.getType() == operation->getResult(0).getType())
        constants.emplace_back(operation, value);
    });
    if (constants.empty())
      return;
    if (!staticDataDirectory.empty())
      if (std::error_code error =
              llvm::sys::fs::create_directories(staticDataDirectory)) {
        module.emitError() << "cannot create static tensor directory "
                           << staticDataDirectory << ": " << error.message();
        return signalPassFailure();
      }

    SmallVector<Attribute> records(existing.begin(), existing.end());
    for (auto [operation, value] : constants) {
      int64_t id = nextId++;
      // Graph-local constants were inline in the IR already, so keeping the
      // value inline costs no extra memory and lets Host finalization fold
      // scalar and shape operands exactly as before.
      auto record = replaceWithStaticTensor(
          module, operation, value, id, ("constant_" + Twine(id)).str(),
          staticDataDirectory, /*keepInlineData=*/true);
      if (failed(record))
        return signalPassFailure();
      records.push_back(*record);
    }
    module->setAttr(onnxStaticTensorsAttr,
                    ArrayAttr::get(module.getContext(), records));
  }
};
} // namespace
} // namespace vrc
