#include "VRC/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <iterator>
#include <utility>

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_ATTACHDEVICERESOURCEOPERANDCONTRACTSPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {

/// Declares the Device buffer operands of the runtime entry points whose
/// buffer arguments the static resource analysis must see: the leading
/// `buffers` arguments are buffers, the last of them written.
void attachDeviceResourceOperandContracts(ModuleOp module) {
  Builder builder(module.getContext());
  struct Contract {
    StringLiteral name;
    unsigned buffers;
  };
  static constexpr Contract known[] = {
      {"__vrc_enqueue_udondnn_large_convolution_device_args", 4},
      {"__vrc_enqueue_udondnn_large_activation_device_args", 3},
      {"__vrc_enqueue_udondnn_large_pooling_device_args", 3},
      {"__vrc_enqueue_udondnn_large_op_tensor_device_args", 4},
      {"__vrc_enqueue_udonaten_arena_rms_norm_reduce_device_args", 2},
      {"__vrc_enqueue_udonaten_arena_rms_norm_finalize_device_args", 4},
      {"__vrc_enqueue_udonaten_argmax_reduce_device_args", 2},
      {"__vrc_enqueue_udonaten_argmax_finalize_device_args", 3},
      {"__vrc_enqueue_udondnn_arena_softmax_device_args", 2},
      {"__vrc_enqueue_udononnx_matmul_nbits_q4_device_args", 5},
      {"__vrc_enqueue_udononnx_unary_device_args", 2},
      {"__vrc_enqueue_udononnx_transpose_device_args", 2},
      {"__vrc_enqueue_udononnx_slice_device_args", 2},
      {"__vrc_enqueue_udononnx_expand_device_args", 2},
      {"__vrc_enqueue_udononnx_concat2_device_args", 3},
      {"__vrc_enqueue_udononnx_gather_axis0_device_args", 3},
      {"__vrc_enqueue_udononnx_cast_i64_f32_device_args", 2},
      {"__vrc_enqueue_udononnx_equal_f32_device_args", 3},
      {"__vrc_enqueue_udononnx_where_f32_device_args", 4},
      {"__vrc_enqueue_udononnx_scatter_nd_f32_device_args", 4},
      {"__vrc_enqueue_udonblas_arena_sgemm_device_args", 3},
      {"__vrc_enqueue_udonblas_arena_sgemm_strided_batched_device_args", 3},
  };
  SmallVector<std::pair<func::FuncOp, ArrayAttr>> contracts;
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    StringRef name = function.getSymName();
    auto contract = llvm::find_if(
        known, [&](const Contract &entry) { return entry.name == name; });
    // The entry point of a user OpenCL C kernel names its buffer count:
    // __vrc_enqueue_user_<id>_<buffers>_<kernel> (OpenCLUserKernels.h).
    unsigned buffers = contract == std::end(known) ? 0 : contract->buffers;
    StringRef user = name;
    if (contract == std::end(known) &&
        user.consume_front("__vrc_enqueue_user_")) {
      StringRef count = user.split('_').second.split('_').first;
      if (count.getAsInteger(10, buffers))
        buffers = 0;
    }
    if (!function.isExternal() || buffers == 0 ||
        function.getNumArguments() <= buffers)
      continue;
    SmallVector<mlir::Attribute> entries;
    for (unsigned index = 1; index <= buffers; ++index)
      entries.push_back(builder.getDictionaryAttr({
          builder.getNamedAttr(
              "access",
              builder.getStringAttr(index == buffers ? "write" : "read")),
          builder.getNamedAttr("capture", builder.getStringAttr("none")),
          builder.getNamedAttr("index", builder.getI32IntegerAttr(index)),
      }));
    contracts.emplace_back(function, builder.getArrayAttr(entries));
  }
  for (auto &[function, contract] : contracts)
    function->setAttr("vrc.device_resource_operands", contract);
}

struct AttachDeviceResourceOperandContractsPass
    : impl::AttachDeviceResourceOperandContractsPassBase<
          AttachDeviceResourceOperandContractsPass> {
  void runOnOperation() override {
    attachDeviceResourceOperandContracts(getOperation());
  }
};

} // namespace
} // namespace vrc
