// Adds the Device buffers of the Host buffer plan of a Unity library to the
// mutable handle capacity of its ONNX Host: the plan names the ONNX base
// capacity and one byte size per allocation of the C Host.
#include "VRC/Conversion/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
using namespace mlir;
namespace vrc {
#define GEN_PASS_DEF_APPLYHOSTBUFFERPLANPASS
#include "VRC/Conversion/Passes.h.inc"
namespace {
constexpr StringLiteral planAttribute = "vrc.host_buffer_plan";

struct ApplyHostBufferPlanPass
    : impl::ApplyHostBufferPlanPassBase<ApplyHostBufferPlanPass> {
  void runOnOperation() override {
    if (failed(apply(getOperation())))
      signalPassFailure();
  }

  LogicalResult apply(ModuleOp module) {
    auto plan = module->getAttrOfType<DictionaryAttr>(planAttribute);
    if (!plan)
      return success();
    auto schema = plan.getAs<IntegerAttr>("schema_version");
    auto base = plan.getAs<IntegerAttr>("base_capacity");
    auto total = plan.getAs<IntegerAttr>("total_capacity");
    auto extra = plan.getAs<ArrayAttr>("extra_buffer_bytes");
    auto capacity =
        module->getAttrOfType<IntegerAttr>("udoncl.mutable_handle_capacity");
    auto bytes =
        module->getAttrOfType<DenseI64ArrayAttr>("udoncl.mutable_buffer_bytes");
    // A Unity library without an ONNX Host has no handles of its own: the
    // plan's base is 0.
    Builder builder(module.getContext());
    if (!capacity && !bytes) {
      capacity = builder.getI64IntegerAttr(0);
      bytes = builder.getDenseI64ArrayAttr({});
    }
    if (!schema || schema.getInt() != 1 || !base || !total || !extra ||
        !capacity || !bytes || capacity.getInt() != base.getInt() ||
        static_cast<int64_t>(bytes.size()) != base.getInt() ||
        total.getInt() != base.getInt() + static_cast<int64_t>(extra.size()))
      return module.emitError(
          "Host buffer plan is inconsistent with ONNX capacity");
    SmallVector<int64_t> combined(bytes.asArrayRef());
    for (Attribute value : extra) {
      auto item = dyn_cast<IntegerAttr>(value);
      if (!item || item.getInt() <= 0)
        return module.emitError(
            "Host buffer plan contains an invalid byte size");
      combined.push_back(item.getInt());
    }
    module->setAttr("udoncl.mutable_buffer_bytes",
                    builder.getDenseI64ArrayAttr(combined));
    module->setAttr("udoncl.mutable_handle_capacity",
                    builder.getI64IntegerAttr(total.getInt()));
    return success();
  }
};
} // namespace
} // namespace vrc
