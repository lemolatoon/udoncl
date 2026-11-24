// Synthesizes the gather kernel of an ONNX Device fusion group and the
// declaration that dispatches it through the generated kernel channel.
#ifndef VRC_CONVERSION_ONNXTOUDONHOST_ONNXHOSTFUSIONKERNEL_H
#define VRC_CONVERSION_ONNXTOUDONHOST_ONNXHOSTFUSIONKERNEL_H

#include "ONNXHostFusion.h"
#include "ONNXHostSequenceExtents.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringMap.h"

#include <functional>
#include <string>

namespace vrc {
struct DeviceRowReduction;

/// How the Host dispatches one fusion group: a call to `callee` with the
/// buffer token of every tensor of `leaves`, in order, then the token of the
/// root's result. Every shape is static, so the kernel takes no scalar but
/// the zero-row extent of a constant result prefix (see dispatch()), which
/// follows the result token, and the offset of a view with a run-time offset
/// (DeviceFusionLeafView), which comes last.
///
/// A two-pass row reduction (deviceSoftmaxSchedule) first calls
/// `statisticsCallee` with the leaf tokens and then a scratch buffer of at
/// least `statisticsWords` words, which it writes; `callee` then takes the
/// leaf tokens, that scratch buffer and the result.
/// A run-time i32 a kernel over tensors with an axis of a bounded dimension
/// takes after its other scalars: the value of bounded dimension `index`
/// (in GraphRecord::dimensionBounds), or the pitch the binding of graph
/// input (`output` false) or output `index` gave its dynamic axis.
struct DeviceFusionScalar {
  enum Kind { Dimension, BoundPitch };
  Kind kind;
  unsigned index;
  bool output = false;
  bool operator==(const DeviceFusionScalar &other) const {
    return kind == other.kind && index == other.index && output == other.output;
  }
};

/// How the buffer that holds a tensor with an axis of a bounded dimension
/// lays out that axis: the pitch of the axis is its run-time extent (a
/// Device buffer written dense), its largest extent (a constant planned at
/// the bound), or the pitch a binding gave (a graph input or output).
struct DeviceFusionLayout {
  enum Kind { Dense, Bound, Binding };
  Kind kind = Dense;
  /// The binding of a Binding layout.
  bool output = false;
  unsigned binding = 0;
  /// The largest pitch: the words of the buffer cover this extent.
  int64_t capacity = 0;
};
using DeviceFusionLayouts = std::function<DeviceFusionLayout(unsigned)>;

struct DeviceFusionDispatch {
  std::string callee;
  llvm::SmallVector<unsigned> leaves;
  std::string statisticsCallee;
  int64_t statisticsWords = 0;
  /// The run-time scalars both kernels take after the others, in order.
  llvm::SmallVector<DeviceFusionScalar> scalars;
};

/// A leaf that the kernel reads in a larger buffer: the element at
/// coordinates c of the leaf tensor is the element at c, with `offset` added
/// on `axis`, of a row-major tensor of `shape`. A banked sequence state reads
/// its carried rows this way from the previous present.
///
/// With `dynamicOffset` the offset is a run-time value instead: the kernel
/// takes it as an i32 after its other scalars and reads with
/// min(value, offset), so `offset` is the largest one a run may pass. A
/// banked state that may commit part of a run
/// (sequenceStatePartialAdvanceStep) reads its carried rows this way.
struct DeviceFusionLeafView {
  unsigned tensor;
  llvm::SmallVector<int64_t> shape;
  int64_t axis;
  int64_t offset;
  bool dynamicOffset = false;
};

/// The synthesized kernels of one Host module. They live in one carrier
/// `gpu.module`; groups with the same expression, shapes and immediates
/// share a kernel.
class DeviceFusionKernels {
public:
  /// `layouts` gives the layout of the buffer of every leaf or root tensor
  /// with an axis of a bounded dimension.
  DeviceFusionKernels(mlir::ModuleOp module, const DeviceFusion &fusion,
                      DeviceFusionLayouts layouts = {})
      : module(module), fusion(fusion), layouts(std::move(layouts)) {}

  /// Returns the dispatch of the group rooted at `rootNode`, synthesizing its
  /// kernel unless an identical one exists. `rootNode` may also be a node
  /// outside every group that its library kernel cannot run: a Softmax whose
  /// schedule has two passes, or a node beyond the limits of its library
  /// kernel (isSynthesizedDeviceNode), which the kernel evaluates on its own.
  ///
  /// With `prefix`, the kernel takes the zero-row extent Z of the prefix's
  /// state as an i32 after the result token and writes the elements of the
  /// root's result whose coordinate on the prefix axis is below Z as
  /// `prefix->bits` instead of evaluating the expression there
  /// (ONNXHostSequenceExtents.h). The statistics pass of a two-pass row
  /// reduction always reads the whole row.
  mlir::FailureOr<DeviceFusionDispatch>
  dispatch(unsigned rootNode, const DeviceConstantPrefix *prefix = nullptr);

  /// The dispatch of the expression of `rootNode` with the leaf `view.tensor`
  /// read through `view`; its buffer argument holds the whole view shape.
  /// `rootNode` may also be a node outside every group, which the kernel then
  /// evaluates on its own. `prefix` is as above.
  mlir::FailureOr<DeviceFusionDispatch>
  dispatch(unsigned rootNode, const DeviceFusionLeafView &view,
           const DeviceConstantPrefix *prefix = nullptr);

  /// The dispatch of the Concat `rootNode` of an in-place Concat pair
  /// (InPlaceConcat in ONNXHostPlan.h) when its first operand is the leading
  /// block of the result's buffer. The kernel neither takes nor reads that
  /// operand: it writes the elements of the second operand, after the first
  /// operand's run-time extent along the axis, and every other fragment
  /// keeps its word (a partial-output kernel whose vrc.output_granule is
  /// whole rows). The other leaves, the result and the scalars are as for
  /// dispatch().
  mlir::FailureOr<DeviceFusionDispatch> dispatchInPlace(unsigned rootNode);

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
             const DeviceRowReduction *schedule, FusedKernelRole role,
             const DeviceConstantPrefix *prefix, bool inPlace = false);

  mlir::ModuleOp module;
  const DeviceFusion &fusion;
  DeviceFusionLayouts layouts;
  mlir::Operation *carrier = nullptr;
  /// Dispatch declaration of every kernel by its structure key.
  llvm::StringMap<std::string> callees;
};
} // namespace vrc

#endif
