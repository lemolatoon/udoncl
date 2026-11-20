#ifndef VRC_CONVERSION_ONNXTOUDONHOST_ONNXHOSTPLAN_H
#define VRC_CONVERSION_ONNXTOUDONHOST_ONNXHOSTPLAN_H

#include "ONNXHostGraph.h"

#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"

#include <optional>

namespace vrc {
struct ONNXHostPlan {
  bool deviceGraph;
  llvm::SmallVector<unsigned> deviceStorage;
  /// Root of the fusion group of every node; see ONNXHostFusion.h.
  llvm::SmallVector<unsigned> deviceFusionRoots;
  llvm::SmallVector<unsigned> ownedDeviceBuffers;
  int64_t operatorScratchBytes;
  llvm::SmallVector<int64_t> mutableBufferBytes;
  int64_t mutableHandleCapacity;
  /// Indices into GraphRecord::sequenceStates of the states that alternate
  /// between two banks instead of being copied on advance; see
  /// sequenceStateBanks().
  llvm::SmallVector<unsigned> sequenceStateBanks;
};

struct ONNXHostDeviceSliceShape {
  llvm::SmallVector<int64_t> starts;
  llvm::SmallVector<int64_t> steps;
};

struct ONNXHostDeviceConcatShape {
  int64_t innerWords;
  int64_t leftExtent;
  int64_t rightExtent;
};

struct ONNXHostDeviceGatherShape {
  int64_t innerWords;
  int64_t axisExtent;
  int64_t indexWords;
};

struct ONNXHostDeviceScatterNDShape {
  int64_t tupleRank;
  int64_t updateCount;
  int64_t sliceWords;
  int64_t indexWords;
};

struct ONNXHostDeviceMatMulShape {
  int64_t m;
  int64_t n;
  int64_t k;
  int64_t leftBatches;
  int64_t rightBatches;
  int64_t outputBatches;
};

using HostPlanRecord = ONNXHostPlan;
using DeviceSliceShape = ONNXHostDeviceSliceShape;
using DeviceConcatShape = ONNXHostDeviceConcatShape;
using DeviceGatherShape = ONNXHostDeviceGatherShape;
using DeviceScatterNDShape = ONNXHostDeviceScatterNDShape;
using DeviceMatMulShape = ONNXHostDeviceMatMulShape;

inline constexpr llvm::StringLiteral onnxHostPlanAttr = "udoncl.onnx_host_plan";
inline constexpr llvm::StringLiteral hostGraphPlanAttr = onnxHostPlanAttr;

llvm::SmallVector<int64_t, 4> tensor4d(mlir::RankedTensorType type);
llvm::SmallVector<int64_t, 4> broadcastTensor4d(mlir::RankedTensorType type);
int64_t integerAttribute(const NodeRecord &node, llvm::StringRef name,
                         int64_t fallback);
llvm::SmallVector<int64_t>
integerArrayAttribute(const NodeRecord &node, llvm::StringRef name,
                      llvm::ArrayRef<int64_t> fallback);
bool isMetadataOnlyAlias(const NodeRecord &node);
bool isDeviceAlias(const GraphRecord &graph, const NodeRecord &node);
std::optional<float> scalarF32(const TensorRecord &tensor);
std::optional<llvm::SmallVector<int64_t>>
staticI64Values(const TensorRecord &tensor);
std::optional<DeviceSliceShape> deviceSliceShape(const GraphRecord &graph,
                                                 const NodeRecord &node);
bool isDeviceExpandNode(const GraphRecord &graph, const NodeRecord &node);
std::optional<DeviceConcatShape> deviceConcatShape(const GraphRecord &graph,
                                                   const NodeRecord &node);
/// A two-input Concat beyond the 16384 words of udononnx_concat2_f32 but
/// within the 131072 words of a synthesized gather kernel, which then runs it
/// even when it is not part of a fusion group.
bool isSynthesizedConcatNode(const GraphRecord &graph, const NodeRecord &node);
std::optional<DeviceGatherShape> deviceGatherShape(const GraphRecord &graph,
                                                   const NodeRecord &node);
bool isDeviceCastI64F32Node(const GraphRecord &graph, const NodeRecord &node);
bool isDeviceEqualNode(const GraphRecord &graph, const NodeRecord &node);
bool isDeviceWhereNode(const GraphRecord &graph, const NodeRecord &node);
std::optional<DeviceScatterNDShape>
deviceScatterNDShape(const GraphRecord &graph, const NodeRecord &node);
bool isDeviceRmsNormNode(const GraphRecord &graph, const NodeRecord &node);
bool isDeviceSoftmaxNode(const GraphRecord &graph, const NodeRecord &node);
std::optional<llvm::SmallVector<int64_t>>
deviceTransposePermutation(const GraphRecord &graph, const NodeRecord &node);
/// K bound of the udonblas_arena_sgemm kernels (32 guarded chunks of 256).
inline constexpr int64_t deviceSgemmMaxK = 8192;
/// Operand words the non-batched udonblas_arena_sgemm kernel addresses; a
/// larger operand goes through the strided-batched kernel with one batch.
inline constexpr int64_t deviceSgemmOperandWords = 16384;
std::optional<DeviceMatMulShape> deviceMatMulShape(const GraphRecord &graph,
                                                   const NodeRecord &node);
bool isDeviceQ4Node(const GraphRecord &graph, const NodeRecord &node);
/// K blocks of 32 that one udononnx_matmul_nbits_q4 fragment accumulates. It
/// must equal UDONONNX_Q4_DISPATCH_BLOCKS in
/// libraries/udononnxoperators/device/matmul_nbits.cl.
inline constexpr int64_t deviceQ4DispatchBlocks = 256;
/// Partial planes one udononnx_matmul_nbits_q4 dispatch writes or sums; it
/// must equal UDONONNX_Q4_MAX_SPLITS.
inline constexpr int64_t deviceQ4MaxSplits = 64;

/// A reduction-heavy Device kernel in which one fragment computes one output
/// word by a serial reduction (a GEMV-shaped kernel). Split-K runs it as S
/// partial fragments per output, each over `unitsPerSplit` consecutive units
/// of the reduction, followed by one ordered sum pass over the S partials.
struct DeviceSplitKProblem {
  /// Output words, i.e. fragments of the unsplit kernel.
  int64_t outputs;
  /// Serial reduction length of one output word, in kernel units.
  int64_t units;
  /// Texel Loads and shader statements one fragment executes per unit.
  int64_t loadsPerUnit;
  int64_t statementsPerUnit;
  /// Kernel limits: units one fragment may reduce, partial planes one
  /// dispatch may write or sum, and words of the partial surface.
  int64_t maxUnitsPerSplit;
  int64_t maxSplits;
  int64_t partialWords;
};
struct DeviceSplitK {
  /// 1 runs the unsplit kernel in one dispatch.
  int64_t splits;
  int64_t unitsPerSplit;
};
/// Chooses the split count with the smallest modelled GPU time; see the
/// definition for the model. Returns nothing when no schedule fits the
/// kernel limits.
std::optional<DeviceSplitK>
chooseDeviceSplitK(const DeviceSplitKProblem &problem);
/// The split-K schedule of a Device Q4 MatMulNBits node (isDeviceQ4Node).
std::optional<DeviceSplitK> deviceQ4SplitK(const GraphRecord &graph,
                                           const NodeRecord &node);

/// A row reduction followed by an elementwise pass over the row (Softmax).
/// The one-pass kernel re-reduces the whole row in every output fragment, so
/// its work grows with the square of the row length. The two-pass schedule
/// instead writes `chunks` partial statistics per row once, each over
/// `chunkLength` consecutive elements, and the elementwise pass combines the
/// partials of its row.
struct DeviceRowReductionProblem {
  /// Rows, and elements of one row (the reduced axis).
  int64_t rows;
  int64_t extent;
  /// Texel Loads and statements of one evaluation of the row operand.
  int64_t loadsPerElement;
  int64_t statementsPerElement;
  /// Words of the statistics surface (two words per partial).
  int64_t statisticsWords;
};
struct DeviceRowReduction {
  /// 0 runs the one-pass kernel.
  int64_t chunks;
  int64_t chunkLength;
};
/// Chooses between the one-pass kernel and the two-pass schedule with the
/// split-K latency/throughput model (chooseDeviceSplitK). Two passes must
/// save more than a tenth of the one-pass time.
DeviceRowReduction
chooseDeviceRowReduction(const DeviceRowReductionProblem &problem);
/// Most partial statistics per row of a two-pass row reduction.
inline constexpr int64_t deviceRowReductionMaxChunks = 64;
/// Words of the operator scratch a statistics pass may write: two rows of
/// the 8192-word surface, so its whole launch surface fits.
inline constexpr int64_t deviceRowStatisticsWords = 16384;
/// Shape of the rows a Device Softmax node (isDeviceSoftmaxNode) normalizes:
/// `outer` x `extent` x `inner`, reducing the middle axis.
struct DeviceSoftmaxRows {
  int64_t outer;
  int64_t extent;
  int64_t inner;
  int64_t rows() const { return outer * inner; }
};
std::optional<DeviceSoftmaxRows> deviceSoftmaxRows(const GraphRecord &graph,
                                                   const NodeRecord &node);
/// The schedule of a Device Softmax node, assuming the largest operand a
/// fused Softmax may read; chunks 0 is the one-pass kernel.
DeviceRowReduction deviceSoftmaxSchedule(const GraphRecord &graph,
                                         const NodeRecord &node);
int64_t operatorScratchBytes(const GraphRecord &graph);
bool needsQ4Surface(const GraphRecord &graph, unsigned tensorIndex);
/// Bytes of the Device buffer that holds `tensorIndex`. The input of a
/// sequence state also holds a whole present tensor: it is one of the two
/// banks of a banked state.
int64_t deviceAllocationBytes(const GraphRecord &graph, unsigned tensorIndex);
/// The sequence states whose carried rows stay where the last run wrote
/// them. Every run's Concat writes its present into one of two banks, the
/// public output buffer or the (present-sized) public input buffer, and reads
/// the previous rows from the other bank through a view, so advancing the
/// state copies nothing. A state qualifies when the graph runs on the Device,
/// the Concat that produces the present is a two-input Concat a fused kernel
/// can evaluate, and nothing but that Concat's first operand reads the past.
llvm::SmallVector<unsigned> sequenceStateBanks(const GraphRecord &graph);
int64_t bindingCapacityBytes(const GraphRecord &graph, unsigned tensorIndex);
bool supportsDeviceNode(const GraphRecord &graph, const NodeRecord &node);
bool supportsDeviceGraph(const GraphRecord &graph);
/// Assigns every tensor the tensor whose Device buffer holds it.
/// `fusionRoots` maps every node to the node whose dispatch computes it: the
/// operands of a fused node stay live until that dispatch, and its result
/// has no buffer of its own.
llvm::SmallVector<unsigned> deviceStorage(const GraphRecord &graph,
                                          llvm::ArrayRef<unsigned> fusionRoots);
llvm::SmallVector<unsigned>
ownedDeviceBuffers(const GraphRecord &graph, bool deviceGraph,
                   llvm::ArrayRef<unsigned> storage);
/// The tensors whose physical buffer token a Device graph run resolves at
/// run time, in resolution order without repeats: the bound inputs, the
/// bound outputs, and the Device buffers the session owns.
llvm::SmallVector<unsigned> resolvedRunTensors(const GraphRecord &graph,
                                               const HostPlanRecord &plan);
llvm::SmallVector<int64_t> mutableBufferBytes(const GraphRecord &graph,
                                              bool deviceGraph,
                                              llvm::ArrayRef<unsigned> storage);
mlir::FailureOr<HostPlanRecord> readHostPlan(mlir::ModuleOp module,
                                             const GraphRecord &graph);
} // namespace vrc

#endif
