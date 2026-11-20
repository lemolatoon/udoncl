#include "ONNXHostFusion.h"
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
  using Base::Base;

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
        InFlightDiagnostic diagnostic =
            module.emitError()
            << "ONNX Host fallback cannot call Device-pointer provider "
            << node.provider << " for " << node.operation;
        const NodeRecord *blocker =
            llvm::find_if(graph->nodes, [&](const NodeRecord &candidate) {
              return !supportsDeviceNode(*graph, candidate);
            });
        Diagnostic &note = diagnostic.attachNote();
        note << "first Device-unsupported node is " << blocker->operation
             << " via " << blocker->provider << " with inputs (";
        llvm::interleaveComma(blocker->inputs, note, [&](int64_t input) {
          if (input < 0)
            note << "none";
          else
            note << graph->tensors[static_cast<unsigned>(input)].type;
        });
        note << ") and result " << graph->tensors[blocker->output].type;
        signalPassFailure();
        return;
      }
    }
    // A Device dispatch can only read storage something has written: a public
    // input bound by the caller, an immutable static tensor, or the result of
    // an earlier node. Any other operand (for example a dense constant that
    // was never externalized) would be backed by an unwritten mutable buffer.
    if (deviceGraph) {
      llvm::SmallDenseSet<unsigned> written(graph->inputs.begin(),
                                            graph->inputs.end());
      for (const NodeRecord &node : graph->nodes) {
        for (int64_t input : node.inputs) {
          if (input < 0)
            continue;
          unsigned tensorIndex = static_cast<unsigned>(input);
          const TensorRecord &tensor = graph->tensors[tensorIndex];
          if (written.contains(tensorIndex) || tensor.staticResourceId >= 0)
            continue;
          module.emitError()
              << "ONNX Device graph node " << node.operation << " via "
              << node.provider << " reads " << tensor.symbol << " ("
              << tensor.type
              << ") that is neither a graph input, an immutable static "
                 "tensor, nor the result of an earlier node";
          signalPassFailure();
          return;
        }
        written.insert(node.output);
      }
    }
    SmallVector<unsigned> fusionRoots =
        DeviceFusion::plan(*graph, fuseDeviceNodes);
    SmallVector<unsigned> storage = deviceStorage(*graph, fusionRoots);
    SmallVector<unsigned> owned =
        ownedDeviceBuffers(*graph, deviceGraph, storage);
    SmallVector<int64_t> bufferBytes =
        mutableBufferBytes(*graph, deviceGraph, storage);
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
    SmallVector<int64_t> fusionRootIds(fusionRoots.begin(), fusionRoots.end());
    SmallVector<int64_t> ownedIds;
    ownedIds.reserve(owned.size());
    for (unsigned value : owned)
      ownedIds.push_back(value);
    SmallVector<int64_t> bankIds;
    if (bankSequenceState)
      for (unsigned value : sequenceStateBanks(*graph))
        bankIds.push_back(value);
    SmallVector<NamedAttribute> planEntries{
        builder.getNamedAttr("device_graph", builder.getBoolAttr(deviceGraph)),
        builder.getNamedAttr("device_fusion_roots",
                             builder.getDenseI64ArrayAttr(fusionRootIds)),
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
    };
    if (!bankIds.empty())
      planEntries.push_back(builder.getNamedAttr(
          "sequence_state_banks", builder.getDenseI64ArrayAttr(bankIds)));
    module->setAttr(hostGraphPlanAttr, builder.getDictionaryAttr(planEntries));
  }
};
} // namespace
} // namespace vrc
