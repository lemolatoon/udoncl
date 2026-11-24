// Derives the Unity buffers of an ONNX Host. The ONNX execution plan fixes
// every Device buffer of the Host: buffer ids 1 to
// `udoncl.mutable_handle_capacity` with the capacities of
// `udoncl.mutable_buffer_bytes`, all private. A Host exposes a graph tensor
// by binding a UDONCL_PUBLIC buffer of its own to it.
#include "VRC/Conversion/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include <map>
#include <string>
using namespace mlir;
namespace vrc {
#define GEN_PASS_DEF_DERIVEONNXUNITYBUFFERSPASS
#include "VRC/Conversion/Passes.h.inc"
namespace {
constexpr StringLiteral descriptionAttribute = "vrc.onnx_unity_buffers";

struct DeriveONNXUnityBuffersPass
    : impl::DeriveONNXUnityBuffersPassBase<DeriveONNXUnityBuffersPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    module->removeAttr(descriptionAttribute);
    if (failed(derive(module)))
      signalPassFailure();
  }

  LogicalResult derive(ModuleOp module) {
    if (!module->hasAttr("udoncl.onnx_binding"))
      return success();
    Builder builder(module.getContext());
    std::map<int64_t, DictionaryAttr> derived;
    auto handleCapacity =
        module->getAttrOfType<IntegerAttr>("udoncl.mutable_handle_capacity");
    auto mutableBufferBytes =
        module->getAttrOfType<DenseI64ArrayAttr>("udoncl.mutable_buffer_bytes");
    if (!handleCapacity || !mutableBufferBytes ||
        handleCapacity.getInt() !=
            static_cast<int64_t>(mutableBufferBytes.size()))
      return module.emitError(
          "ONNX mutable handle capacity is inconsistent with its buffers");
    for (int64_t id = 1; id <= handleCapacity.getInt(); ++id) {
      int64_t allocationBytes = mutableBufferBytes.asArrayRef()[id - 1];
      if (allocationBytes <= 0)
        return module.emitError(
            "generated ONNX buffer has a non-positive capacity");
      std::string name = "private_" + std::to_string(id);
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
