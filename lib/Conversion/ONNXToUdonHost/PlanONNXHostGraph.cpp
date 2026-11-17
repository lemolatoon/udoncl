#include "ONNXHostGraph.h"
#include "VRC/Conversion/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_PLANONNXHOSTGRAPHPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {
struct PlanONNXHostGraphPass
    : impl::PlanONNXHostGraphPassBase<PlanONNXHostGraphPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    if (failed(planONNXHostGraph(getOperation())))
      signalPassFailure();
  }
};
} // namespace
} // namespace vrc
