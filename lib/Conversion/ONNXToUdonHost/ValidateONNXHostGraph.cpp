#include "ONNXHostGraph.h"
#include "VRC/Conversion/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_VALIDATEONNXHOSTGRAPHPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {
struct ValidateONNXHostGraphPass
    : impl::ValidateONNXHostGraphPassBase<ValidateONNXHostGraphPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    if (failed(validateONNXHostGraph(getOperation())))
      signalPassFailure();
  }
};
} // namespace
} // namespace vrc
