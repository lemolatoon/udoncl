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
    ModuleOp module = getOperation();
    if (failed(readONNXHostGraph(module))) {
      signalPassFailure();
      return;
    }
    module->setAttr(hostGraphValidatedAttr, UnitAttr::get(module.getContext()));
  }
};
} // namespace
} // namespace vrc
