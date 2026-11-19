#include "ONNXStaticTensor.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "src/Dialect/ONNX/ElementsAttr/DisposableElementsAttr.hpp"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"

using namespace mlir;

namespace {
StringRef initializerName(Location location) {
  if (auto named = dyn_cast<NameLoc>(location)) {
    StringRef name = named.getName().strref();
    if (name.starts_with("Initializer_"))
      return name.drop_front(StringRef("Initializer_").size());
    return initializerName(named.getChildLoc());
  }
  if (auto fused = dyn_cast<FusedLoc>(location))
    for (Location nested : fused.getLocations())
      if (StringRef name = initializerName(nested); !name.empty())
        return name;
  if (auto callSite = dyn_cast<CallSiteLoc>(location)) {
    if (StringRef name = initializerName(callSite.getCallee()); !name.empty())
      return name;
    return initializerName(callSite.getCaller());
  }
  return {};
}

struct ExternalizeONNXInitializersPass
    : PassWrapper<ExternalizeONNXInitializersPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExternalizeONNXInitializersPass)

  ExternalizeONNXInitializersPass() = default;
  ExternalizeONNXInitializersPass(const ExternalizeONNXInitializersPass &pass)
      : PassWrapper<ExternalizeONNXInitializersPass,
                    OperationPass<ModuleOp>>() {
    staticDataDirectory = pass.staticDataDirectory.getValue();
  }

  Pass::Option<std::string> staticDataDirectory{
      *this, "static-data-directory",
      llvm::cl::desc("Directory receiving external static tensor payloads"),
      llvm::cl::init("")};

  StringRef getArgument() const final {
    return "udoncl-externalize-onnx-initializers";
  }
  StringRef getDescription() const final {
    return "Replace imported ONNX initializers with immutable static tensor "
           "symbols";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (!staticDataDirectory.empty()) {
      std::error_code error =
          llvm::sys::fs::create_directories(staticDataDirectory);
      if (error) {
        module.emitError() << "cannot create static tensor directory "
                           << staticDataDirectory << ": " << error.message();
        signalPassFailure();
        return;
      }
    }
    SmallVector<Operation *> initializers;
    module.walk([&](Operation *operation) {
      if (operation->getName().getStringRef() == "onnx.Constant" &&
          !initializerName(operation->getLoc()).empty())
        initializers.push_back(operation);
    });

    SmallVector<Attribute> records;
    records.reserve(initializers.size());
    for (auto [index, operation] : llvm::enumerate(initializers)) {
      if (operation->getNumResults() != 1) {
        operation->emitError("ONNX initializer must have one result");
        signalPassFailure();
        return;
      }
      DenseElementsAttr value;
      if (auto dense = operation->getAttrOfType<DenseElementsAttr>("value"))
        value = dense;
      else if (auto disposable =
                   operation->getAttrOfType<DisposableElementsAttr>("value"))
        value = disposable.toDenseElementsAttr();
      if (!vrc::isStaticTensorRepresentable(value)) {
        operation->emitError(
            "ONNX initializer has an unsupported static tensor representation");
        signalPassFailure();
        return;
      }
      // Finalization needs scalar/shape constants, but carrying model weights
      // through another MLIR parse multiplies peak memory. Large payloads live
      // only in the sidecar directory and are validated by length and SHA-256
      // when the Unity artifact is packed.
      size_t bytes = value.getType().getNumElements() *
                     (value.getElementType().getIntOrFloatBitWidth() / 8);
      auto record = vrc::replaceWithStaticTensor(
          module, operation, value, index, initializerName(operation->getLoc()),
          staticDataDirectory, bytes <= 4096);
      if (failed(record)) {
        signalPassFailure();
        return;
      }
      records.push_back(*record);
    }
    module->setAttr(vrc::onnxStaticTensorsAttr,
                    ArrayAttr::get(module.getContext(), records));
  }
};

static PassRegistration<ExternalizeONNXInitializersPass> registration;
} // namespace
