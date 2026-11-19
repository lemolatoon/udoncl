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
  llvm::SmallVector<unsigned> ownedDeviceBuffers;
  int64_t operatorScratchBytes;
  llvm::SmallVector<int64_t> mutableBufferBytes;
  int64_t mutableHandleCapacity;
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
/// K blocks of 32 that one udononnx_matmul_nbits_q4 dispatch accumulates. It
/// must equal UDONONNX_Q4_DISPATCH_BLOCKS in
/// libraries/udononnxoperators/device/matmul_nbits.cl; a longer K is split
/// into ceil(blocks / deviceQ4DispatchBlocks) dispatches at compile time.
inline constexpr int64_t deviceQ4DispatchBlocks = 256;
int64_t operatorScratchBytes(const GraphRecord &graph);
bool needsQ4Surface(const GraphRecord &graph, unsigned tensorIndex);
int64_t deviceAllocationBytes(const GraphRecord &graph, unsigned tensorIndex);
int64_t bindingCapacityBytes(const GraphRecord &graph, unsigned tensorIndex);
bool supportsDeviceNode(const GraphRecord &graph, const NodeRecord &node);
bool supportsDeviceGraph(const GraphRecord &graph);
llvm::SmallVector<unsigned> deviceStorage(const GraphRecord &graph);
llvm::SmallVector<unsigned>
ownedDeviceBuffers(const GraphRecord &graph, bool deviceGraph,
                   llvm::ArrayRef<unsigned> storage);
llvm::SmallVector<int64_t> mutableBufferBytes(const GraphRecord &graph,
                                              bool deviceGraph);
mlir::FailureOr<HostPlanRecord> readHostPlan(mlir::ModuleOp module,
                                             const GraphRecord &graph);
} // namespace vrc

#endif
