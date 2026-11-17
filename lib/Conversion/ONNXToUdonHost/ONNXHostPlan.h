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
std::optional<DeviceMatMulShape> deviceMatMulShape(const GraphRecord &graph,
                                                   const NodeRecord &node);
bool isDeviceQ4Node(const GraphRecord &graph, const NodeRecord &node);
int64_t operatorScratchBytes(const GraphRecord &graph);
bool needsQ4Surface(const GraphRecord &graph, unsigned tensorIndex);
int64_t deviceAllocationBytes(const GraphRecord &graph, unsigned tensorIndex);
bool supportsDeviceNode(const GraphRecord &graph, const NodeRecord &node);
bool supportsDeviceGraph(const GraphRecord &graph);
llvm::SmallVector<unsigned> deviceStorage(const GraphRecord &graph);
llvm::SmallVector<int64_t> mutableBufferBytes(const GraphRecord &graph,
                                              bool deviceGraph);
mlir::FailureOr<HostPlanRecord> readHostPlan(mlir::ModuleOp module,
                                             const GraphRecord &graph);
} // namespace vrc

#endif
