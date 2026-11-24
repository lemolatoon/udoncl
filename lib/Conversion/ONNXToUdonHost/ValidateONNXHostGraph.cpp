#include "ONNXHostGraph.h"
#include "ONNXHostPlan.h"
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
    FailureOr<GraphRecord> graph = readONNXHostGraph(module);
    if (failed(graph)) {
      signalPassFailure();
      return;
    }
    // Every node that reads or writes a tensor with an axis of a bounded
    // dimension must support its run-time extent.
    for (const NodeRecord &node : graph->nodes)
      if (std::optional<std::string> reason =
              unsupportedDynamicExtent(*graph, node)) {
        emitError(node.location.value_or(module.getLoc())) << *reason;
        signalPassFailure();
        return;
      }
    module->setAttr(hostGraphValidatedAttr, UnitAttr::get(module.getContext()));
  }
};
} // namespace
} // namespace vrc
