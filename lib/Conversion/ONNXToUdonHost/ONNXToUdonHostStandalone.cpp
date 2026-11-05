#include "ONNXToUdonHost.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;

namespace vrc {
namespace {
struct StandaloneONNXToUdonHostPass
    : PassWrapper<StandaloneONNXToUdonHostPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StandaloneONNXToUdonHostPass)
  StringRef getArgument() const final { return "udoncl-lower-onnx-to-host"; }
  StringRef getDescription() const final {
    return "Lower ONNX dialect operations to Udon library host calls";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect>();
  }
  void runOnOperation() override {
    if (failed(lowerONNXToUdonHost(getOperation())))
      signalPassFailure();
  }
};
} // namespace

static PassRegistration<StandaloneONNXToUdonHostPass> registration;
} // namespace vrc
