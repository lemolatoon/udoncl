// Derives the Unity buffers of an ONNX Host. The ONNX execution plan fixes
// every Device buffer of the Host: buffer ids 1 to
// `udoncl.mutable_handle_capacity` with the capacities of
// `udoncl.mutable_buffer_bytes`. With public tensor buffers, the graph inputs
// and then the outputs, in graph order, are buffers 1 to N, named and typed
// after the ONNX binding; every other buffer is private.
#include "VRC/Conversion/Passes.h"
#include "VRC/Conversion/UnityBufferNames.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include <map>
#include <set>
#include <string>
using namespace mlir;
namespace vrc {
#define GEN_PASS_DEF_DERIVEONNXUNITYBUFFERSPASS
#include "VRC/Conversion/Passes.h.inc"
namespace {
constexpr StringLiteral descriptionAttribute = "vrc.onnx_unity_buffers";

struct DeriveONNXUnityBuffersPass
    : impl::DeriveONNXUnityBuffersPassBase<DeriveONNXUnityBuffersPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    module->removeAttr(descriptionAttribute);
    if (failed(derive(module)))
      signalPassFailure();
  }

  LogicalResult derive(ModuleOp module) {
    auto binding = module->getAttrOfType<DictionaryAttr>("udoncl.onnx_binding");
    if (!binding) {
      if (publicTensorBuffers)
        return module.emitError("--onnx-public-tensor-buffers "
                                "(ONNX_PUBLIC_TENSOR_BUFFERS) requires an "
                                "ONNX Host");
      return success();
    }
    auto inputs = binding.getAs<ArrayAttr>("inputs");
    auto outputs = binding.getAs<ArrayAttr>("outputs");
    if (!inputs || !outputs)
      return module.emitError("ONNX binding has no input/output metadata");

    Builder builder(module.getContext());
    std::map<int64_t, DictionaryAttr> derived;
    std::set<std::string> names;
    int64_t nextId = 1;
    auto append = [&](ArrayAttr values, bool output) -> LogicalResult {
      for (Attribute value : values) {
        auto entry = dyn_cast<DictionaryAttr>(value);
        auto index = entry ? entry.getAs<IntegerAttr>("index") : IntegerAttr();
        auto name = entry ? entry.getAs<StringAttr>("name") : StringAttr();
        auto dataType =
            entry ? entry.getAs<StringAttr>("data_type") : StringAttr();
        auto logicalBytes =
            entry ? entry.getAs<IntegerAttr>("logical_bytes") : IntegerAttr();
        auto capacityBytes =
            entry ? entry.getAs<IntegerAttr>("capacity_bytes") : IntegerAttr();
        if (!index || !name || !dataType || !logicalBytes || !capacityBytes ||
            logicalBytes.getInt() < 0 || capacityBytes.getInt() <= 0 ||
            logicalBytes.getInt() > capacityBytes.getInt() ||
            (dataType.getValue() != "float32" &&
             dataType.getValue() != "int32" && dataType.getValue() != "int64"))
          return module.emitError(
              "ONNX public tensor has an unsupported binding ABI");
        std::string bufferName = onnxPublicBufferName(name.getValue(), output,
                                                      index.getInt(), names);
        if (bufferName.empty())
          return module.emitError(
              "ONNX public tensor names are not uniquely mappable");
        int64_t id = nextId++;
        derived[id] = builder.getDictionaryAttr({
            builder.getNamedAttr(
                "byte_size", builder.getI64IntegerAttr(capacityBytes.getInt())),
            builder.getNamedAttr("element_type", dataType),
            builder.getNamedAttr("id", builder.getI64IntegerAttr(id)),
            builder.getNamedAttr(
                "logical_byte_size",
                builder.getI64IntegerAttr(logicalBytes.getInt())),
            builder.getNamedAttr("name", builder.getStringAttr(bufferName)),
            builder.getNamedAttr(
                "public_access",
                builder.getStringAttr(output ? "output" : "input")),
            builder.getNamedAttr("public_index",
                                 builder.getI64IntegerAttr(index.getInt())),
        });
      }
      return success();
    };
    if (publicTensorBuffers)
      if (failed(append(inputs, false)) || failed(append(outputs, true)))
        return failure();

    auto handleCapacity =
        module->getAttrOfType<IntegerAttr>("udoncl.mutable_handle_capacity");
    auto mutableBufferBytes =
        module->getAttrOfType<DenseI64ArrayAttr>("udoncl.mutable_buffer_bytes");
    if (!handleCapacity || !mutableBufferBytes ||
        handleCapacity.getInt() !=
            static_cast<int64_t>(mutableBufferBytes.size()) ||
        handleCapacity.getInt() < nextId - 1)
      return module.emitError(
          "ONNX mutable handle capacity is inconsistent with I/O");
    for (int64_t id = 1; id <= handleCapacity.getInt(); ++id) {
      int64_t allocationBytes = mutableBufferBytes.asArrayRef()[id - 1];
      if (allocationBytes <= 0)
        return module.emitError(
            "generated ONNX buffer has a non-positive capacity");
      auto publicBuffer = derived.find(id);
      if (publicBuffer != derived.end()) {
        if (publicBuffer->second.getAs<IntegerAttr>("byte_size").getInt() !=
            allocationBytes)
          return module.emitError() << "generated ONNX public buffer " << id
                                    << " has inconsistent physical capacity";
        continue;
      }
      std::string name = "private_" + std::to_string(id);
      if (!names.insert(name).second)
        return module.emitError("generated ONNX private buffer name collision");
      derived[id] = builder.getDictionaryAttr({
          builder.getNamedAttr("byte_size",
                               builder.getI64IntegerAttr(allocationBytes)),
          builder.getNamedAttr("id", builder.getI64IntegerAttr(id)),
          builder.getNamedAttr("name", builder.getStringAttr(name)),
      });
    }
    SmallVector<Attribute> buffers;
    for (auto &[id, buffer] : derived)
      buffers.push_back(buffer);
    module->setAttr(descriptionAttribute, builder.getArrayAttr(buffers));
    return success();
  }
};
} // namespace
} // namespace vrc
