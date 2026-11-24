#include "ONNXHostABI.h"
#include "ONNXHostGraph.h"
#include "ONNXHostMaterialization.h"
#include "ONNXHostPlan.h"
#include "VRC/Conversion/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringMap.h"

#include <utility>

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_ATTACHONNXHOSTMETADATAPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {
void attachBindingMetadata(ModuleOp module, const GraphRecord &graph) {
  Builder builder(module.getContext());
  auto ioMetadata = [&](ArrayRef<unsigned> tensors) {
    SmallVector<Attribute> entries;
    entries.reserve(tensors.size());
    for (auto [index, tensorIndex] : llvm::enumerate(tensors)) {
      const TensorRecord &tensor = graph.tensors[tensorIndex];
      entries.push_back(builder.getDictionaryAttr({
          builder.getNamedAttr("index", builder.getI32IntegerAttr(index)),
          builder.getNamedAttr("name", builder.getStringAttr(tensor.name)),
          builder.getNamedAttr("data_type",
                               builder.getStringAttr(onnxHostDataTypeName(
                                   tensor.type.getElementType()))),
          builder.getNamedAttr(
              "rank", builder.getI32IntegerAttr(tensor.type.getRank())),
          builder.getNamedAttr("dimensions", builder.getDenseI64ArrayAttr(
                                                 tensor.type.getShape())),
          builder.getNamedAttr(
              "logical_bytes",
              builder.getI64IntegerAttr(
                  tensor.type.getNumElements() *
                  onnxHostElementByteSize(tensor.type.getElementType()))),
          builder.getNamedAttr("capacity_bytes",
                               builder.getI64IntegerAttr(
                                   bindingCapacityBytes(graph, tensorIndex))),
      }));
    }
    return builder.getArrayAttr(entries);
  };
  SmallVector<NamedAttribute> fields{
      builder.getNamedAttr("abi", builder.getStringAttr("udononnx-runtime-v3")),
      builder.getNamedAttr("model_index", builder.getI32IntegerAttr(0)),
      builder.getNamedAttr("model_name", builder.getStringAttr("model_0")),
      builder.getNamedAttr("inputs", ioMetadata(graph.inputs)),
      builder.getNamedAttr("outputs", ioMetadata(graph.outputs)),
  };
  // Transitional (removed in stage 3): a library built with
  // TRANSITIONAL_SEQUENCE_STATE names the symbols of the private binding
  // FinalizeONNXHost generates for it, the sequence-state ones when the graph
  // has sequence state.
  if (module->hasAttr("udoncl.onnx_sequence_states")) {
    SmallVector<NamedAttribute> symbols;
    auto symbol = [&](StringRef key, StringRef name) {
      symbols.push_back(builder.getNamedAttr(key, builder.getStringAttr(name)));
    };
    symbol("create_symbol", "__udononnx_transitional_create");
    symbol("bind_all_symbol", "__udononnx_transitional_bind_all");
    symbol("resolve_input_symbol", "__udononnx_transitional_resolve_input");
    symbol("resolve_output_symbol", "__udononnx_transitional_resolve_output");
    symbol("enqueue_run_symbol", "__udononnx_transitional_enqueue_run");
    if (!graph.sequenceStates.empty()) {
      symbol("reset_sequence_state_symbol",
             "__udononnx_transitional_reset_sequence_state");
      symbol("advance_sequence_state_symbol",
             "__udononnx_transitional_advance_sequence_state");
    }
    symbol("dispose_symbol", "__udononnx_transitional_dispose");
    fields.push_back(builder.getNamedAttr("transitional_sequence_state",
                                          builder.getDictionaryAttr(symbols)));
  }
  module->setAttr("udoncl.onnx_binding", builder.getDictionaryAttr(fields));
}

void attachDeviceGraphMetadata(ModuleOp module, const GraphRecord &graph,
                               bool deviceGraph) {
  Builder builder(module.getContext());
  llvm::StringMap<int64_t> counts;
  llvm::StringMap<int64_t> unsupported;
  for (const NodeRecord &node : graph.nodes) {
    ++counts[node.operation];
    if (!supportsDeviceNode(graph, node))
      ++unsupported[node.operation + "\n" + node.provider];
  }
  SmallVector<NamedAttribute> countAttributes;
  for (const auto &entry : counts)
    countAttributes.push_back(builder.getNamedAttr(
        entry.first(), builder.getI64IntegerAttr(entry.second)));
  llvm::sort(countAttributes, [](NamedAttribute left, NamedAttribute right) {
    return left.getName().strref() < right.getName().strref();
  });
  SmallVector<Attribute> unsupportedAttributes;
  for (const auto &entry : unsupported) {
    StringRef key = entry.first();
    size_t separator = key.find('\n');
    unsupportedAttributes.push_back(builder.getDictionaryAttr({
        builder.getNamedAttr("operation",
                             builder.getStringAttr(key.take_front(separator))),
        builder.getNamedAttr(
            "provider", builder.getStringAttr(key.drop_front(separator + 1))),
        builder.getNamedAttr("count", builder.getI64IntegerAttr(entry.second)),
    }));
  }
  llvm::sort(unsupportedAttributes, [](Attribute left, Attribute right) {
    auto lhs = cast<DictionaryAttr>(left);
    auto rhs = cast<DictionaryAttr>(right);
    return std::make_pair(lhs.getAs<StringAttr>("operation").getValue(),
                          lhs.getAs<StringAttr>("provider").getValue()) <
           std::make_pair(rhs.getAs<StringAttr>("operation").getValue(),
                          rhs.getAs<StringAttr>("provider").getValue());
  });
  module->setAttr("udoncl.onnx_device_graph", builder.getBoolAttr(deviceGraph));
  module->setAttr("udoncl.onnx_operator_counts",
                  builder.getDictionaryAttr(countAttributes));
  module->setAttr("udoncl.onnx_device_unsupported",
                  builder.getArrayAttr(unsupportedAttributes));
}

struct AttachONNXHostMetadataPass
    : impl::AttachONNXHostMetadataPassBase<AttachONNXHostMetadataPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (module->hasAttr(onnxHostMetadataAttachedAttr)) {
      module.emitError("ONNX Host metadata has already been attached");
      signalPassFailure();
      return;
    }
    if (!module->hasAttr(hostGraphValidatedAttr)) {
      module.emitError("ONNX Host metadata requires graph validation");
      signalPassFailure();
      return;
    }
    if (!module->hasAttr(hostGraphPlanAttr)) {
      module.emitError("ONNX Host metadata requires graph planning");
      signalPassFailure();
      return;
    }
    auto graph = readONNXHostGraph(module);
    if (failed(graph)) {
      signalPassFailure();
      return;
    }
    auto plan = readHostPlan(module, *graph);
    if (failed(plan)) {
      signalPassFailure();
      return;
    }
    MLIRContext *context = module.getContext();
    attachBindingMetadata(module, *graph);
    attachDeviceGraphMetadata(module, *graph, plan->deviceGraph);
    module->setAttr("udoncl.mutable_handle_capacity",
                    IntegerAttr::get(IntegerType::get(context, 64),
                                     plan->mutableHandleCapacity));
    module->setAttr("udoncl.mutable_buffer_bytes",
                    DenseI64ArrayAttr::get(context, plan->mutableBufferBytes));
    module->setAttr("udoncl.onnx_lowering",
                    StringAttr::get(context, "library-calls-v2"));
    module->setAttr(onnxHostMetadataAttachedAttr, UnitAttr::get(context));
  }
};
} // namespace
} // namespace vrc
