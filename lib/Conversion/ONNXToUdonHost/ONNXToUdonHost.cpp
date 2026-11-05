#include "ONNXToUdonHost.h"
#include "VRC/Conversion/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_LOWERONNXTOUDONHOSTPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {
struct LowerONNXToUdonHostPass
    : impl::LowerONNXToUdonHostPassBase<LowerONNXToUdonHostPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect>();
  }
  void runOnOperation() override {
    if (failed(lowerONNXToUdonHost(getOperation())))
      signalPassFailure();
  }
};
} // namespace
} // namespace vrc
