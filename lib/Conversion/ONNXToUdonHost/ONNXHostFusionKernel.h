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
struct DeviceRowReduction;

/// How the Host dispatches one fusion group: a call to `callee` with the
/// buffer token of every tensor of `leaves`, in order, then the token of the
/// root's result. The kernel takes no scalars because every shape is static.
///
/// A two-pass row reduction (deviceSoftmaxSchedule) first calls
/// `statisticsCallee` with the leaf tokens and then a scratch buffer of at
/// least `statisticsWords` words, which it writes; `callee` then takes the
/// leaf tokens, that scratch buffer and the result.
struct DeviceFusionDispatch {
  std::string callee;
  llvm::SmallVector<unsigned> leaves;
  std::string statisticsCallee;
  int64_t statisticsWords = 0;
};

/// A leaf that the kernel reads in a larger buffer: the element at
/// coordinates c of the leaf tensor is the element at c, with `offset` added
/// on `axis`, of a row-major tensor of `shape`. A banked sequence state reads
/// its carried rows this way from the previous present.
struct DeviceFusionLeafView {
  unsigned tensor;
  llvm::SmallVector<int64_t> shape;
  int64_t axis;
  int64_t offset;
};

/// The synthesized kernels of one Host module. They live in one carrier
/// `gpu.module`; groups with the same expression, shapes and immediates
/// share a kernel.
class DeviceFusionKernels {
public:
  DeviceFusionKernels(mlir::ModuleOp module, const DeviceFusion &fusion)
      : module(module), fusion(fusion) {}

  /// Returns the dispatch of the group rooted at `rootNode`, synthesizing its
  /// kernel unless an identical one exists. `rootNode` may also be a node
  /// outside every group that its library kernel cannot run: a Softmax whose
  /// schedule has two passes, or a node beyond the limits of its library
  /// kernel (isSynthesizedDeviceNode), which the kernel evaluates on its own.
  mlir::FailureOr<DeviceFusionDispatch> dispatch(unsigned rootNode);

  /// The dispatch of the expression of `rootNode` with the leaf `view.tensor`
  /// read through `view`; its buffer argument holds the whole view shape.
  /// `rootNode` may also be a node outside every group, which the kernel then
  /// evaluates on its own.
  mlir::FailureOr<DeviceFusionDispatch>
  dispatch(unsigned rootNode, const DeviceFusionLeafView &view);

private:
  enum class FusedKernelRole {
    /// The root's value at every element.
    Whole,
    /// The partial statistics of a two-pass row reduction.
    RowStatistics,
    /// The root's value at every element from those statistics.
    FromRowStatistics,
  };
  mlir::FailureOr<DeviceFusionDispatch>
  synthesize(unsigned rootNode, const DeviceFusionLeafView *view,
             const DeviceRowReduction *schedule, FusedKernelRole role);

  mlir::ModuleOp module;
  const DeviceFusion &fusion;
  mlir::Operation *carrier = nullptr;
  /// Dispatch declaration of every kernel by its structure key.
  llvm::StringMap<std::string> callees;
};
} // namespace vrc

#endif
