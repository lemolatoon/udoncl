#include "ONNXHostGraph.h"
#include "ONNXHostPlan.h"
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
    ModuleOp module = getOperation();
    if (!module->hasAttr(hostGraphValidatedAttr)) {
      module.emitError("ONNX Host planning requires the graph validation pass");
      signalPassFailure();
      return;
    }
    auto graph = readONNXHostGraph(module);
    if (failed(graph)) {
      signalPassFailure();
      return;
    }
    bool deviceGraph = supportsDeviceGraph(*graph);
    if (!deviceGraph) {
      for (const NodeRecord &node : graph->nodes) {
        if (!StringRef(node.provider).starts_with("udonblas"))
          continue;
        module.emitError()
            << "ONNX Host fallback cannot call Device-pointer provider "
            << node.provider << " for " << node.operation;
        signalPassFailure();
        return;
      }
    }
    SmallVector<unsigned> storage = deviceStorage(*graph);
    SmallVector<unsigned> owned =
        ownedDeviceBuffers(*graph, deviceGraph, storage);
    SmallVector<int64_t> bufferBytes = mutableBufferBytes(*graph, deviceGraph);
    int64_t handleCapacity = bufferBytes.size();
    constexpr int64_t maximumMutableHandleCapacity = 512;
    if (handleCapacity <= 0 || handleCapacity > maximumMutableHandleCapacity) {
      module.emitError()
          << "ONNX graph requires " << handleCapacity
          << " simultaneously live mutable handles; backend limit is "
          << maximumMutableHandleCapacity;
      signalPassFailure();
      return;
    }

    Builder builder(module.getContext());
    SmallVector<int64_t> storageIds;
    storageIds.reserve(storage.size());
    for (unsigned value : storage)
      storageIds.push_back(value);
    SmallVector<int64_t> ownedIds;
    ownedIds.reserve(owned.size());
    for (unsigned value : owned)
      ownedIds.push_back(value);
    module->setAttr(
        hostGraphPlanAttr,
        builder.getDictionaryAttr({
            builder.getNamedAttr("device_graph",
                                 builder.getBoolAttr(deviceGraph)),
            builder.getNamedAttr("device_storage",
                                 builder.getDenseI64ArrayAttr(storageIds)),
            builder.getNamedAttr("owned_device_buffers",
                                 builder.getDenseI64ArrayAttr(ownedIds)),
            builder.getNamedAttr(
                "operator_scratch_bytes",
                builder.getI64IntegerAttr(operatorScratchBytes(*graph))),
            builder.getNamedAttr("mutable_buffer_bytes",
                                 builder.getDenseI64ArrayAttr(bufferBytes)),
            builder.getNamedAttr("mutable_handle_capacity",
                                 builder.getI64IntegerAttr(handleCapacity)),
        }));
  }
};
} // namespace
} // namespace vrc
