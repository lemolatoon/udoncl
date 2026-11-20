// Synthesizes the gather kernel of an ONNX Device fusion group and the
// declaration that dispatches it through the generated kernel channel.
#ifndef VRC_CONVERSION_ONNXTOUDONHOST_ONNXHOSTFUSIONKERNEL_H
#define VRC_CONVERSION_ONNXTOUDONHOST_ONNXHOSTFUSIONKERNEL_H

#include "ONNXHostFusion.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringMap.h"

#include <string>

namespace vrc {
/// How the Host dispatches one fusion group: a call to `callee` with the
/// buffer token of every tensor of `leaves`, in order, then the token of the
/// root's result. The kernel takes no scalars because every shape is static.
struct DeviceFusionDispatch {
  std::string callee;
  llvm::SmallVector<unsigned> leaves;
};

/// The synthesized kernels of one Host module. They live in one carrier
/// `gpu.module`; groups with the same expression, shapes and immediates
/// share a kernel.
class DeviceFusionKernels {
public:
  DeviceFusionKernels(mlir::ModuleOp module, const DeviceFusion &fusion)
      : module(module), fusion(fusion) {}

  /// Returns the dispatch of the group rooted at `rootNode`, synthesizing its
  /// kernel unless an identical one exists.
  mlir::FailureOr<DeviceFusionDispatch> dispatch(unsigned rootNode);

private:
  mlir::ModuleOp module;
  const DeviceFusion &fusion;
  mlir::Operation *carrier = nullptr;
  /// Dispatch declaration of every kernel by its structure key.
  llvm::StringMap<std::string> callees;
};
} // namespace vrc

#endif
