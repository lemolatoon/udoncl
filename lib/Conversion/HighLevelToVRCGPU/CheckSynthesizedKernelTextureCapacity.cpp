// Checks that the texture shape the Device buffers of a Unity library share
// can hold the output of every compiler-synthesized kernel. A synthesized
// kernel writes vrc.words words into its output buffer; when every Unity
// buffer is a texture of one shape, that shape must hold all of them.
#include "VRC/Conversion/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "llvm/ADT/STLExtras.h"
using namespace mlir;
namespace vrc {
#define GEN_PASS_DEF_CHECKSYNTHESIZEDKERNELTEXTURECAPACITYPASS
#include "VRC/Conversion/Passes.h.inc"
namespace {
struct CheckSynthesizedKernelTextureCapacityPass
    : impl::CheckSynthesizedKernelTextureCapacityPassBase<
          CheckSynthesizedKernelTextureCapacityPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (bufferWidths.size() != bufferHeights.size()) {
      module.emitError("buffer-widths and buffer-heights must list the same "
                       "number of Unity buffers");
      return signalPassFailure();
    }
    // Only a texture shape every buffer shares bounds a kernel's output.
    if (bufferWidths.empty() ||
        !llvm::all_of(llvm::zip(bufferWidths, bufferHeights), [&](auto shape) {
          return std::get<0>(shape) == bufferWidths.front() &&
                 std::get<1>(shape) == bufferHeights.front();
        }))
      return;
    int64_t width = bufferWidths.front();
    int64_t height = bufferHeights.front();
    WalkResult result = module.walk([&](gpu::GPUFuncOp kernel) {
      auto words = kernel->getAttrOfType<IntegerAttr>("vrc.words");
      if (!kernel->hasAttr("vrc.synthesized_kernel") || !words ||
          width * height >= words.getInt())
        return WalkResult::advance();
      kernel.emitError("synthesized kernel @")
          << kernel.getName() << " writes " << words.getInt()
          << " words but the Unity buffers are " << width << "x" << height
          << " textures";
      return WalkResult::interrupt();
    });
    if (result.wasInterrupted())
      signalPassFailure();
  }
};
} // namespace
} // namespace vrc
