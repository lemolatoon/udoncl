// Describes the `onnx_binding` entry of a Unity library manifest. The ONNX
// Host metadata (`udoncl.onnx_binding`, attached by
// udoncl-attach-onnx-host-metadata) names the model, its graph inputs and
// outputs and the symbols of the generated binding; this pass checks it and
// records the manifest entry in its serialized form.
#include "VRC/Conversion/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
using namespace mlir;
namespace vrc {
#define GEN_PASS_DEF_DESCRIBEONNXBINDINGMANIFESTPASS
#include "VRC/Conversion/Passes.h.inc"
namespace {
constexpr StringLiteral bindingAttribute = "udoncl.onnx_binding";
constexpr StringLiteral descriptionAttribute = "vrc.onnx_binding_manifest";

struct DescribeONNXBindingManifestPass
    : impl::DescribeONNXBindingManifestPassBase<
          DescribeONNXBindingManifestPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    module->removeAttr(descriptionAttribute);
    if (failed(describe(module)))
      signalPassFailure();
  }

  /// The manifest entries of the graph inputs or outputs `values`.
  FailureOr<ArrayAttr> describeTensors(ModuleOp module, ArrayAttr values) {
    Builder builder(module.getContext());
    SmallVector<Attribute> result;
    for (Attribute value : values) {
      auto entry = dyn_cast<DictionaryAttr>(value);
      auto index = entry ? entry.getAs<IntegerAttr>("index") : IntegerAttr();
      auto name = entry ? entry.getAs<StringAttr>("name") : StringAttr();
      auto dataType =
          entry ? entry.getAs<StringAttr>("data_type") : StringAttr();
      auto rank = entry ? entry.getAs<IntegerAttr>("rank") : IntegerAttr();
      auto dimensions = entry ? entry.getAs<DenseI64ArrayAttr>("dimensions")
                              : DenseI64ArrayAttr();
      auto logicalBytes =
          entry ? entry.getAs<IntegerAttr>("logical_bytes") : IntegerAttr();
      auto capacityBytes =
          entry ? entry.getAs<IntegerAttr>("capacity_bytes") : IntegerAttr();
      if (!index || !name || !dataType || !rank || !dimensions ||
          !logicalBytes || !capacityBytes)
        return module.emitError("ONNX binding tensor metadata is incomplete");
      SmallVector<Attribute> shape;
      for (int64_t dimension : dimensions.asArrayRef())
        shape.push_back(builder.getI64IntegerAttr(dimension));
      // A dynamic axis is -1, with its largest extent and the name of its
      // bounded dimension.
      auto bounds = entry.getAs<DenseI64ArrayAttr>("dimension_bounds");
      auto names = entry.getAs<ArrayAttr>("dimension_names");
      bool dynamic = llvm::is_contained(dimensions.asArrayRef(), -1);
      if (dynamic != static_cast<bool>(bounds) || (bounds && !names) ||
          (bounds && (bounds.size() != dimensions.size() ||
                      names.size() != dimensions.size())))
        return module.emitError(
            "ONNX binding tensor bounded dimensions are malformed");
      SmallVector<NamedAttribute> fields({
          builder.getNamedAttr("index",
                               builder.getI64IntegerAttr(index.getInt())),
          builder.getNamedAttr("name", name),
          builder.getNamedAttr("data_type", dataType),
          builder.getNamedAttr("rank",
                               builder.getI64IntegerAttr(rank.getInt())),
          builder.getNamedAttr("dimensions", builder.getArrayAttr(shape)),
          builder.getNamedAttr("logical_bytes", builder.getI64IntegerAttr(
                                                    logicalBytes.getInt())),
          builder.getNamedAttr("capacity_bytes", builder.getI64IntegerAttr(
                                                     capacityBytes.getInt())),
      });
      if (bounds) {
        SmallVector<Attribute> extents;
        for (int64_t bound : bounds.asArrayRef())
          extents.push_back(builder.getI64IntegerAttr(bound));
        fields.push_back(builder.getNamedAttr("dimension_bounds",
                                              builder.getArrayAttr(extents)));
        fields.push_back(builder.getNamedAttr("dimension_names", names));
      }
      result.push_back(builder.getDictionaryAttr(fields));
    }
    return builder.getArrayAttr(result);
  }

  LogicalResult describe(ModuleOp module) {
    auto binding = module->getAttrOfType<DictionaryAttr>(bindingAttribute);
    if (!binding)
      return success();
    auto abi = binding.getAs<StringAttr>("abi");
    auto modelIndex = binding.getAs<IntegerAttr>("model_index");
    auto modelName = binding.getAs<StringAttr>("model_name");
    auto inputs = binding.getAs<ArrayAttr>("inputs");
    auto outputs = binding.getAs<ArrayAttr>("outputs");
    auto transitional =
        binding.getAs<DictionaryAttr>("transitional_sequence_state");
    if (!abi || !modelIndex || !modelName || !inputs || !outputs)
      return module.emitError("ONNX binding metadata is incomplete");
    auto describedInputs = describeTensors(module, inputs);
    if (failed(describedInputs))
      return failure();
    auto describedOutputs = describeTensors(module, outputs);
    if (failed(describedOutputs))
      return failure();

    Builder builder(module.getContext());
    SmallVector<NamedAttribute> fields{
        builder.getNamedAttr("abi", abi),
        builder.getNamedAttr("model_index",
                             builder.getI64IntegerAttr(modelIndex.getInt())),
        builder.getNamedAttr("model_name", modelName),
        builder.getNamedAttr("inputs", *describedInputs),
        builder.getNamedAttr("outputs", *describedOutputs),
    };
    // Transitional (removed in stage 3): only a library built with
    // TRANSITIONAL_SEQUENCE_STATE names the symbols of its private binding,
    // the sequence-state ones when the graph has sequence state.
    if (transitional) {
      SmallVector<NamedAttribute> symbols;
      for (StringRef name :
           {"create_symbol", "bind_all_symbol", "resolve_input_symbol",
            "resolve_output_symbol", "enqueue_run_symbol", "dispose_symbol"}) {
        auto value = transitional.getAs<StringAttr>(name);
        if (!value || value.getValue().empty())
          return module.emitError()
                 << "ONNX binding metadata has no transitional " << name;
        symbols.push_back(builder.getNamedAttr(name, value));
      }
      for (StringRef name :
           {"reset_sequence_state_symbol", "advance_sequence_state_symbol"})
        if (auto value = transitional.getAs<StringAttr>(name);
            value && !value.getValue().empty())
          symbols.push_back(builder.getNamedAttr(name, value));
      fields.push_back(builder.getNamedAttr(
          "transitional_sequence_state", builder.getDictionaryAttr(symbols)));
    }
    module->setAttr(descriptionAttribute, builder.getDictionaryAttr(fields));
    return success();
  }
};
} // namespace
} // namespace vrc
