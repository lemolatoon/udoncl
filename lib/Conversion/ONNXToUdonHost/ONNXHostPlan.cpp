#include "ONNXHostPlan.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstring>
#include <numeric>

using namespace mlir;

namespace vrc {
namespace {
unsigned planElementByteSize(Type type) {
  if (type.isInteger(1) || type.isInteger(8))
    return 1;
  if (type.isF16() || type.isBF16())
    return 2;
  if (type.isInteger(32) || type.isF32())
    return 4;
  if (type.isInteger(64))
    return 8;
  return 0;
}
} // namespace

SmallVector<int64_t, 4> tensor4d(RankedTensorType type) {
  SmallVector<int64_t, 4> shape(4, 1);
  ArrayRef<int64_t> source = type.getShape();
  if (source.size() == 1) {
    shape[1] = source[0];
  } else if (source.size() > 4) {
    shape[0] = 1;
    for (int64_t dimension : source.drop_back(3))
      shape[0] *= dimension;
    llvm::copy(source.take_back(3), shape.begin() + 1);
  } else {
    for (auto [index, dimension] : llvm::enumerate(source))
      shape[index] = dimension;
  }
  return shape;
}

SmallVector<int64_t, 4> broadcastTensor4d(RankedTensorType type) {
  SmallVector<int64_t, 4> shape(4, 1);
  ArrayRef<int64_t> source = type.getShape();
  if (source.size() > 4) {
    shape[0] = 1;
    for (int64_t dimension : source.drop_back(3))
      shape[0] *= dimension;
    llvm::copy(source.take_back(3), shape.begin() + 1);
  } else {
    llvm::copy(source, shape.end() - source.size());
  }
  return shape;
}

int64_t integerAttribute(const NodeRecord &node, StringRef name,
                         int64_t fallback) {
  if (!node.attributes)
    return fallback;
  auto value = node.attributes.getAs<IntegerAttr>(name);
  return value ? value.getInt() : fallback;
}

SmallVector<int64_t> integerArrayAttribute(const NodeRecord &node,
                                           StringRef name,
                                           ArrayRef<int64_t> fallback) {
  if (!node.attributes)
    return SmallVector<int64_t>(fallback);
  Attribute attribute = node.attributes.get(name);
  if (auto dense = dyn_cast_or_null<DenseI64ArrayAttr>(attribute))
    return SmallVector<int64_t>(dense.asArrayRef());
  SmallVector<int64_t> result;
  if (auto array = dyn_cast_or_null<ArrayAttr>(attribute))
    for (Attribute value : array)
      result.push_back(cast<IntegerAttr>(value).getInt());
  return result.empty() ? SmallVector<int64_t>(fallback) : result;
}

bool isMetadataOnlyAlias(const NodeRecord &node) {
  return node.operation == "Flatten" || node.operation == "Reshape" ||
         node.operation == "Identity" || node.operation == "Squeeze" ||
         node.operation == "Unsqueeze";
}

bool isDeviceAlias(const GraphRecord &graph, const NodeRecord &node) {
  if (!isMetadataOnlyAlias(node) || node.inputs.empty() || node.inputs[0] < 0)
    return false;
  const TensorRecord &input = graph.tensors[node.inputs[0]];
  const TensorRecord &output = graph.tensors[node.output];
  return input.type.getElementType() == output.type.getElementType() &&
         input.type.getNumElements() == output.type.getNumElements();
}

constexpr int64_t q4SurfaceWords = 131072;
constexpr int64_t q4SurfaceBytes = q4SurfaceWords * 4;
constexpr int64_t rmsScratchBytes = 16384 * 4;

std::optional<float> scalarF32(const TensorRecord &tensor) {
  if (!tensor.initialValue || !tensor.type.getElementType().isF32() ||
      tensor.type.getNumElements() != 1)
    return std::nullopt;
  auto values = tensor.initialValue.getValues<APFloat>();
  APFloat value = *values.begin();
  return value.convertToFloat();
}

std::optional<SmallVector<int64_t>>
staticI64Values(const TensorRecord &tensor) {
  if (!tensor.initialValue || !tensor.type.getElementType().isInteger(64))
    return std::nullopt;
  SmallVector<int64_t> result;
  result.reserve(tensor.type.getNumElements());
  for (APInt value : tensor.initialValue.getValues<APInt>())
    result.push_back(value.getSExtValue());
  return result;
}

std::optional<DeviceSliceShape> deviceSliceShape(const GraphRecord &graph,
                                                 const NodeRecord &node) {
  if (node.operation != "Slice" || node.provider != "udononnxSlice" ||
      node.inputs.size() < 3 || node.inputs[0] < 0 || node.inputs[1] < 0 ||
      node.inputs[2] < 0)
    return std::nullopt;
  const TensorRecord &input = graph.tensors[node.inputs[0]];
  const TensorRecord &output = graph.tensors[node.output];
  int64_t rank = input.type.getRank();
  if (!input.type.getElementType().isF32() ||
      !output.type.getElementType().isF32() || rank < 1 || rank > 5 ||
      output.type.getRank() != rank || input.type.getNumElements() < 1 ||
      input.type.getNumElements() > 16384 || output.type.getNumElements() < 1 ||
      output.type.getNumElements() > 16384)
    return std::nullopt;
  auto starts = staticI64Values(graph.tensors[node.inputs[1]]);
  auto ends = staticI64Values(graph.tensors[node.inputs[2]]);
  if (!starts || !ends || starts->size() != ends->size() || starts->empty() ||
      starts->size() > static_cast<size_t>(rank))
    return std::nullopt;
  SmallVector<int64_t> axes;
  if (node.inputs.size() >= 4 && node.inputs[3] >= 0) {
    auto values = staticI64Values(graph.tensors[node.inputs[3]]);
    if (!values || values->size() != starts->size())
      return std::nullopt;
    axes = std::move(*values);
  } else {
    for (int64_t axis = 0; axis < static_cast<int64_t>(starts->size()); ++axis)
      axes.push_back(axis);
  }
  SmallVector<int64_t> specifiedSteps(starts->size(), 1);
  if (node.inputs.size() >= 5 && node.inputs[4] >= 0) {
    auto values = staticI64Values(graph.tensors[node.inputs[4]]);
    if (!values || values->size() != starts->size())
      return std::nullopt;
    specifiedSteps = std::move(*values);
  }
  DeviceSliceShape result{SmallVector<int64_t>(rank, 0),
                          SmallVector<int64_t>(rank, 1)};
  SmallVector<bool> sliced(rank, false);
  for (auto [startValue, endValue, axisValue, stepValue] :
       llvm::zip_equal(*starts, *ends, axes, specifiedSteps)) {
    int64_t axis = axisValue < 0 ? axisValue + rank : axisValue;
    if (axis < 0 || axis >= rank || sliced[axis] || !stepValue)
      return std::nullopt;
    sliced[axis] = true;
    int64_t extent = input.type.getDimSize(axis);
    int64_t start = startValue;
    int64_t end = endValue;
    int64_t expected;
    if (stepValue > 0) {
      start = start < 0 ? (start < -extent ? 0 : start + extent)
                        : std::min(start, extent);
      end =
          end < 0 ? (end < -extent ? 0 : end + extent) : std::min(end, extent);
      expected = end > start ? (end - start + stepValue - 1) / stepValue : 0;
    } else {
      int64_t magnitude = -stepValue;
      start = start < 0 ? (start < -extent ? -1 : start + extent)
                        : std::min(start, extent - 1);
      if (end == std::numeric_limits<int64_t>::min())
        end = -1;
      else
        end = end < 0 ? (end < -extent ? -1 : end + extent)
                      : std::min(end, extent - 1);
      expected = start > end ? (start - end + magnitude - 1) / magnitude : 0;
    }
    if (expected != output.type.getDimSize(axis) || start < 0 ||
        start > 65535 || stepValue < 1 || stepValue > 65535)
      return std::nullopt;
    result.starts[axis] = start;
    result.steps[axis] = stepValue;
  }
  for (int64_t axis = 0; axis < rank; ++axis)
    if ((!sliced[axis] &&
         input.type.getDimSize(axis) != output.type.getDimSize(axis)) ||
        input.type.getDimSize(axis) < 1 ||
        input.type.getDimSize(axis) > 65535 ||
        output.type.getDimSize(axis) < 1 ||
        output.type.getDimSize(axis) > 65535)
      return std::nullopt;
  return result;
}

bool isDeviceExpandNode(const GraphRecord &graph, const NodeRecord &node) {
  if (node.operation != "Expand" || node.provider != "udononnxExpand" ||
      node.inputs.empty() || node.inputs[0] < 0)
    return false;
  const TensorRecord &input = graph.tensors[node.inputs[0]];
  const TensorRecord &output = graph.tensors[node.output];
  int64_t inputRank = input.type.getRank();
  int64_t outputRank = output.type.getRank();
  if (!input.type.getElementType().isF32() ||
      !output.type.getElementType().isF32() || inputRank < 0 || inputRank > 5 ||
      outputRank < std::max<int64_t>(inputRank, 1) || outputRank > 5 ||
      input.type.getNumElements() < 1 || input.type.getNumElements() > 16384 ||
      output.type.getNumElements() < 1 || output.type.getNumElements() > 131072)
    return false;
  int64_t rankOffset = outputRank - inputRank;
  for (int64_t outputAxis = 0; outputAxis < outputRank; ++outputAxis) {
    int64_t outputExtent = output.type.getDimSize(outputAxis);
    if (outputExtent < 1 || outputExtent > 65535)
      return false;
    int64_t inputAxis = outputAxis - rankOffset;
    if (inputAxis < 0)
      continue;
    int64_t inputExtent = input.type.getDimSize(inputAxis);
    if (inputExtent < 1 || inputExtent > 65535 ||
        (inputExtent != 1 && inputExtent != outputExtent))
      return false;
  }
  return true;
}

std::optional<DeviceConcatShape> deviceConcatShape(const GraphRecord &graph,
                                                   const NodeRecord &node) {
  if (node.operation != "Concat" || node.provider != "udononnxConcat" ||
      node.inputs.size() != 2 || node.inputs[0] < 0 || node.inputs[1] < 0)
    return std::nullopt;
  const TensorRecord &left = graph.tensors[node.inputs[0]];
  const TensorRecord &right = graph.tensors[node.inputs[1]];
  const TensorRecord &output = graph.tensors[node.output];
  int64_t rank = output.type.getRank();
  int64_t axis = integerAttribute(node, "axis", 0);
  if (axis < 0)
    axis += rank;
  if (!left.type.getElementType().isF32() ||
      !right.type.getElementType().isF32() ||
      !output.type.getElementType().isF32() || rank < 1 || rank > 5 ||
      left.type.getRank() != rank || right.type.getRank() != rank || axis < 0 ||
      axis >= rank || left.type.getNumElements() < 1 ||
      left.type.getNumElements() > 16384 || right.type.getNumElements() < 1 ||
      right.type.getNumElements() > 16384 || output.type.getNumElements() < 1 ||
      output.type.getNumElements() > 16384)
    return std::nullopt;
  int64_t innerWords = 1;
  for (int64_t dimension = 0; dimension < rank; ++dimension) {
    int64_t leftExtent = left.type.getDimSize(dimension);
    int64_t rightExtent = right.type.getDimSize(dimension);
    int64_t outputExtent = output.type.getDimSize(dimension);
    if (leftExtent < 1 || leftExtent > 65535 || rightExtent < 1 ||
        rightExtent > 65535 || outputExtent < 1 || outputExtent > 65535 ||
        (dimension == axis
             ? outputExtent != leftExtent + rightExtent
             : leftExtent != outputExtent || rightExtent != outputExtent))
      return std::nullopt;
    if (dimension > axis)
      innerWords *= outputExtent;
  }
  return DeviceConcatShape{innerWords, left.type.getDimSize(axis),
                           right.type.getDimSize(axis)};
}

std::optional<DeviceGatherShape> deviceGatherShape(const GraphRecord &graph,
                                                   const NodeRecord &node) {
  if (node.operation != "Gather" || node.provider != "udononnxGather" ||
      node.inputs.size() != 2 || node.inputs[0] < 0 || node.inputs[1] < 0)
    return std::nullopt;
  const TensorRecord &data = graph.tensors[node.inputs[0]];
  const TensorRecord &indices = graph.tensors[node.inputs[1]];
  const TensorRecord &output = graph.tensors[node.output];
  int64_t axis = integerAttribute(node, "axis", 0);
  if (axis < 0)
    axis += data.type.getRank();
  if (!data.type.getElementType().isF32() ||
      !output.type.getElementType().isF32() || data.staticResourceId < 0 ||
      axis != 0 || data.type.getRank() < 1 || data.type.getRank() > 5 ||
      (!indices.type.getElementType().isInteger(32) &&
       !indices.type.getElementType().isInteger(64)) ||
      output.type.getNumElements() < 1 ||
      output.type.getNumElements() > 16384 || data.type.getDimSize(0) < 1 ||
      data.type.getDimSize(0) > std::numeric_limits<int32_t>::max())
    return std::nullopt;
  int64_t innerWords = 1;
  for (int64_t extent : data.type.getShape().drop_front()) {
    if (extent < 1 || innerWords > 16384 / extent)
      return std::nullopt;
    innerWords *= extent;
  }
  if (indices.type.getNumElements() < 1 ||
      indices.type.getNumElements() > 16384 / innerWords ||
      output.type.getNumElements() !=
          indices.type.getNumElements() * innerWords)
    return std::nullopt;
  SmallVector<int64_t> expectedShape(indices.type.getShape());
  llvm::append_range(expectedShape, data.type.getShape().drop_front());
  if (output.type.getShape() != ArrayRef<int64_t>(expectedShape))
    return std::nullopt;
  return DeviceGatherShape{innerWords, data.type.getDimSize(0),
                           indices.type.getElementType().isInteger(64) ? 2 : 1};
}

bool isDeviceCastI64F32Node(const GraphRecord &graph, const NodeRecord &node) {
  if (node.operation != "Cast" || node.provider != "udononnxCast" ||
      node.inputs.size() != 1 || node.inputs[0] < 0)
    return false;
  const TensorRecord &input = graph.tensors[node.inputs[0]];
  const TensorRecord &output = graph.tensors[node.output];
  return input.type.getElementType().isInteger(64) &&
         output.type.getElementType().isF32() &&
         input.type.getShape() == output.type.getShape() &&
         output.type.getNumElements() >= 1 &&
         output.type.getNumElements() <= 16384;
}

bool isDeviceBroadcast(ArrayRef<const TensorRecord *> inputs,
                       const TensorRecord &output) {
  int64_t outputRank = output.type.getRank();
  if (outputRank < 0 || outputRank > 4 || output.type.getNumElements() < 1 ||
      output.type.getNumElements() > 16384)
    return false;
  for (int64_t extent : output.type.getShape())
    if (extent < 1 || extent > 65535)
      return false;
  for (const TensorRecord *input : inputs) {
    int64_t inputRank = input->type.getRank();
    if (inputRank < 0 || inputRank > outputRank ||
        input->type.getNumElements() < 1 ||
        input->type.getNumElements() > 16384)
      return false;
    int64_t offset = outputRank - inputRank;
    for (int64_t axis = 0; axis < inputRank; ++axis) {
      int64_t inputExtent = input->type.getDimSize(axis);
      int64_t outputExtent = output.type.getDimSize(axis + offset);
      if (inputExtent < 1 || inputExtent > 65535 ||
          (inputExtent != 1 && inputExtent != outputExtent))
        return false;
    }
  }
  return true;
}

bool isDeviceEqualNode(const GraphRecord &graph, const NodeRecord &node) {
  if (node.operation != "Equal" || node.provider != "udononnxEqual" ||
      node.inputs.size() != 2 || node.inputs[0] < 0 || node.inputs[1] < 0)
    return false;
  const TensorRecord &left = graph.tensors[node.inputs[0]];
  const TensorRecord &right = graph.tensors[node.inputs[1]];
  const TensorRecord &output = graph.tensors[node.output];
  return left.type.getElementType().isF32() &&
         right.type.getElementType().isF32() &&
         output.type.getElementType().isInteger(1) &&
         isDeviceBroadcast({&left, &right}, output);
}

bool isDeviceWhereNode(const GraphRecord &graph, const NodeRecord &node) {
  if (node.operation != "Where" || node.provider != "udononnxWhere" ||
      node.inputs.size() != 3 || node.inputs[0] < 0 || node.inputs[1] < 0 ||
      node.inputs[2] < 0)
    return false;
  const TensorRecord &condition = graph.tensors[node.inputs[0]];
  const TensorRecord &whenTrue = graph.tensors[node.inputs[1]];
  const TensorRecord &whenFalse = graph.tensors[node.inputs[2]];
  const TensorRecord &output = graph.tensors[node.output];
  return condition.type.getElementType().isInteger(1) &&
         whenTrue.type.getElementType().isF32() &&
         whenFalse.type.getElementType().isF32() &&
         output.type.getElementType().isF32() &&
         isDeviceBroadcast({&condition, &whenTrue, &whenFalse}, output);
}

std::optional<DeviceScatterNDShape>
deviceScatterNDShape(const GraphRecord &graph, const NodeRecord &node) {
  if (node.operation != "ScatterND" || node.provider != "udononnxScatterND" ||
      node.inputs.size() != 3 || node.inputs[0] < 0 || node.inputs[1] < 0 ||
      node.inputs[2] < 0)
    return std::nullopt;
  const TensorRecord &data = graph.tensors[node.inputs[0]];
  const TensorRecord &indices = graph.tensors[node.inputs[1]];
  const TensorRecord &updates = graph.tensors[node.inputs[2]];
  const TensorRecord &output = graph.tensors[node.output];
  auto reduction = node.attributes
                       ? node.attributes.getAs<StringAttr>("reduction")
                       : StringAttr();
  int64_t rank = data.type.getRank();
  int64_t indicesRank = indices.type.getRank();
  int64_t tupleRank = indicesRank ? indices.type.getShape().back() : 0;
  if (!data.type.getElementType().isF32() || data.type != output.type ||
      !updates.type.getElementType().isF32() || rank < 1 || rank > 4 ||
      indicesRank < 1 || tupleRank < 1 || tupleRank > rank ||
      (!indices.type.getElementType().isInteger(32) &&
       !indices.type.getElementType().isInteger(64)) ||
      data.type.getNumElements() < 1 || data.type.getNumElements() > 16384 ||
      (reduction && reduction.getValue() != "none"))
    return std::nullopt;
  int64_t updateCount = indices.type.getNumElements() / tupleRank;
  int64_t sliceWords = 1;
  for (int64_t extent : data.type.getShape().drop_front(tupleRank))
    sliceWords *= extent;
  if (updateCount < 1 || updateCount > 256 || sliceWords < 1 ||
      updateCount > 16384 / sliceWords ||
      updates.type.getNumElements() != updateCount * sliceWords)
    return std::nullopt;
  SmallVector<int64_t> expectedUpdates(indices.type.getShape().drop_back());
  llvm::append_range(expectedUpdates,
                     data.type.getShape().drop_front(tupleRank));
  if (updates.type.getShape() != ArrayRef<int64_t>(expectedUpdates))
    return std::nullopt;
  for (int64_t extent : data.type.getShape())
    if (extent < 1 || extent > 65535)
      return std::nullopt;
  return DeviceScatterNDShape{tupleRank, updateCount, sliceWords,
                              indices.type.getElementType().isInteger(64) ? 2
                                                                          : 1};
}

bool isDeviceRmsNormNode(const GraphRecord &graph, const NodeRecord &node) {
  if (node.operation != "RmsNorm" || node.provider != "udonatenRmsNorm" ||
      node.inputs.size() != 3 || node.inputs[0] < 0 || node.inputs[1] < 0 ||
      node.inputs[2] < 0)
    return false;
  const TensorRecord &input = graph.tensors[node.inputs[0]];
  const TensorRecord &weight = graph.tensors[node.inputs[1]];
  const TensorRecord &epsilon = graph.tensors[node.inputs[2]];
  const TensorRecord &output = graph.tensors[node.output];
  int64_t words = input.type.getNumElements();
  int64_t normalizedWords = weight.type.getNumElements();
  std::optional<float> epsilonValue = scalarF32(epsilon);
  if (!input.type.getElementType().isF32() ||
      !weight.type.getElementType().isF32() || input.type != output.type ||
      weight.type.getRank() != 1 || weight.staticResourceId < 0 ||
      !epsilonValue || *epsilonValue < 0.0f || words < 1 || words > 16384 ||
      normalizedWords < 1 || normalizedWords > 4096 ||
      words % normalizedWords != 0 ||
      input.type.getShape().back() != normalizedWords)
    return false;
  int64_t rows = words / normalizedWords;
  int64_t partialsPerRow = (normalizedWords + 63) / 64;
  return rows * partialsPerRow <= 16384;
}

static bool hasDeviceRmsNorm(const GraphRecord &graph) {
  return llvm::any_of(graph.nodes, [&](const NodeRecord &node) {
    return isDeviceRmsNormNode(graph, node);
  });
}

bool isDeviceSoftmaxNode(const GraphRecord &graph, const NodeRecord &node) {
  if (node.operation != "Softmax" || node.provider != "udondnnSoftmaxForward" ||
      node.inputs.size() != 1 || node.inputs[0] < 0)
    return false;
  const TensorRecord &input = graph.tensors[node.inputs[0]];
  const TensorRecord &output = graph.tensors[node.output];
  int64_t rank = input.type.getRank();
  int64_t axis = integerAttribute(node, "axis", -1);
  if (axis < 0)
    axis += rank;
  return input.type.getElementType().isF32() && input.type == output.type &&
         axis >= 0 && axis < rank && input.type.getNumElements() >= 1 &&
         input.type.getNumElements() <= 16384 &&
         input.type.getDimSize(axis) >= 1 && input.type.getDimSize(axis) <= 512;
}

std::optional<SmallVector<int64_t>>
deviceTransposePermutation(const GraphRecord &graph, const NodeRecord &node) {
  if (node.operation != "Transpose" || node.provider != "udononnxTranspose" ||
      node.inputs.size() != 1 || node.inputs[0] < 0)
    return std::nullopt;
  const TensorRecord &input = graph.tensors[node.inputs[0]];
  const TensorRecord &output = graph.tensors[node.output];
  int64_t rank = input.type.getRank();
  if (!input.type.getElementType().isF32() || rank < 1 || rank > 5 ||
      output.type.getRank() != rank || input.type.getNumElements() < 1 ||
      input.type.getNumElements() > 131072 ||
      input.type.getNumElements() != output.type.getNumElements())
    return std::nullopt;
  SmallVector<int64_t> permutation = integerArrayAttribute(node, "perm", {});
  if (permutation.empty())
    for (int64_t axis = rank; axis-- > 0;)
      permutation.push_back(axis);
  if (permutation.size() != static_cast<size_t>(rank))
    return std::nullopt;
  SmallVector<bool> seen(rank, false);
  for (auto [outputAxis, inputAxis] : llvm::enumerate(permutation)) {
    if (inputAxis < 0 || inputAxis >= rank || seen[inputAxis] ||
        input.type.getDimSize(inputAxis) < 1 ||
        input.type.getDimSize(inputAxis) > 65535 ||
        output.type.getDimSize(outputAxis) != input.type.getDimSize(inputAxis))
      return std::nullopt;
    seen[inputAxis] = true;
  }
  return permutation;
}

std::optional<DeviceMatMulShape> deviceMatMulShape(const GraphRecord &graph,
                                                   const NodeRecord &node) {
  if (node.operation != "MatMul" || node.provider != "udonblasSgemm" ||
      node.inputs.size() != 2 || node.inputs[0] < 0 || node.inputs[1] < 0)
    return std::nullopt;
  const TensorRecord &leftTensor = graph.tensors[node.inputs[0]];
  const TensorRecord &rightTensor = graph.tensors[node.inputs[1]];
  const TensorRecord &outputTensor = graph.tensors[node.output];
  ArrayRef<int64_t> left = leftTensor.type.getShape();
  ArrayRef<int64_t> right = rightTensor.type.getShape();
  ArrayRef<int64_t> output = outputTensor.type.getShape();
  if (!leftTensor.type.getElementType().isF32() ||
      !rightTensor.type.getElementType().isF32() ||
      !outputTensor.type.getElementType().isF32() || left.size() < 2 ||
      right.size() < 2 || output.size() < 2 ||
      left.back() != right[right.size() - 2])
    return std::nullopt;
  int64_t m = left[left.size() - 2];
  int64_t k = left.back();
  int64_t n = right.back();
  if (m < 1 || n < 1 || k < 1 || k > deviceSgemmMaxK ||
      output[output.size() - 2] != m || output.back() != n ||
      leftTensor.type.getNumElements() > 131072 ||
      rightTensor.type.getNumElements() > 131072 ||
      outputTensor.type.getNumElements() > 16384)
    return std::nullopt;
  int64_t leftBatches = leftTensor.type.getNumElements() / (m * k);
  int64_t rightBatches = rightTensor.type.getNumElements() / (k * n);
  int64_t outputBatches = outputTensor.type.getNumElements() / (m * n);
  if (outputBatches < 1 || (leftBatches != 1 && leftBatches != outputBatches) ||
      (rightBatches != 1 && rightBatches != outputBatches))
    return std::nullopt;
  return DeviceMatMulShape{m, n, k, leftBatches, rightBatches, outputBatches};
}

bool isDeviceQ4Node(const GraphRecord &graph, const NodeRecord &node) {
  if (node.operation != "MatMulNBits" ||
      node.provider != "udononnxMatMulNBits" || node.inputs.size() < 3 ||
      node.inputs[0] < 0 || node.inputs[1] < 0 || node.inputs[2] < 0 ||
      llvm::any_of(
          ArrayRef<int64_t>(node.inputs).drop_front(3),
          [](int64_t input) { return input >= 0; }))
    return false;
  const TensorRecord &activation = graph.tensors[node.inputs[0]];
  const TensorRecord &packed = graph.tensors[node.inputs[1]];
  const TensorRecord &scales = graph.tensors[node.inputs[2]];
  const TensorRecord &output = graph.tensors[node.output];
  int64_t k = integerAttribute(node, "K", 0);
  int64_t n = integerAttribute(node, "N", 0);
  int64_t blockSize = integerAttribute(node, "block_size", 0);
  int64_t bits = integerAttribute(node, "bits", 0);
  if (!activation.type.getElementType().isF32() ||
      !packed.type.getElementType().isInteger(8) ||
      !scales.type.getElementType().isF32() ||
      !output.type.getElementType().isF32() || packed.staticResourceId < 0 ||
      scales.staticResourceId < 0 || k < 1 || k > 5632 || n < 1 || n > 32000 ||
      blockSize != 32 || bits != 4 || activation.type.getNumElements() % k)
    return false;
  int64_t m = activation.type.getNumElements() / k;
  int64_t blocks = (k + 31) / 32;
  return m >= 1 && m <= 4 && m * n <= q4SurfaceWords &&
         output.type.getNumElements() == m * n &&
         packed.type.getNumElements() == n * blocks * 16 &&
         scales.type.getNumElements() == n * blocks;
}

static bool hasDeviceQ4(const GraphRecord &graph) {
  return llvm::any_of(graph.nodes, [&](const NodeRecord &node) {
    return isDeviceQ4Node(graph, node);
  });
}

int64_t operatorScratchBytes(const GraphRecord &graph) {
  int64_t bytes = hasDeviceQ4(graph) ? q4SurfaceBytes : 0;
  if (hasDeviceRmsNorm(graph))
    bytes = std::max(bytes, rmsScratchBytes);
  for (const NodeRecord &node : graph.nodes)
    if (node.operation == "Gemm" && node.inputs.size() >= 3 &&
        node.inputs[2] >= 0)
      bytes = std::max(bytes, deviceAllocationBytes(graph, node.output));
  return bytes;
}

bool needsQ4Surface(const GraphRecord &graph, unsigned tensorIndex) {
  return llvm::any_of(graph.nodes, [&](const NodeRecord &node) {
    return isDeviceQ4Node(graph, node) && node.output == tensorIndex;
  });
}

int64_t deviceAllocationBytes(const GraphRecord &graph, unsigned tensorIndex) {
  const TensorRecord &tensor = graph.tensors[tensorIndex];
  int64_t elementBytes =
      tensor.type.getElementType().isInteger(1)
          ? 4
          : planElementByteSize(tensor.type.getElementType());
  int64_t bytes = tensor.type.getNumElements() * elementBytes;
  return needsQ4Surface(graph, tensorIndex) ? std::max(bytes, q4SurfaceBytes)
                                            : bytes;
}

bool supportsDeviceNode(const GraphRecord &graph, const NodeRecord &node) {
  auto isF32 = [&](int64_t index) {
    return index >= 0 && static_cast<size_t>(index) < graph.tensors.size() &&
           graph.tensors[static_cast<unsigned>(index)]
               .type.getElementType()
               .isF32();
  };
  auto hasF32InputsAndOutput = [&](unsigned inputCount) {
    return node.inputs.size() >= inputCount &&
           llvm::all_of(ArrayRef<int64_t>(node.inputs).take_front(inputCount),
                        isF32) &&
           isF32(node.output);
  };
  auto supportsLargePointwise = [&]() {
    if (!hasF32InputsAndOutput(2))
      return false;
    SmallVector<int64_t, 4> output =
        broadcastTensor4d(graph.tensors[node.output].type);
    if (graph.tensors[node.output].type.getNumElements() > 131072)
      return false;
    for (int64_t inputIndex : ArrayRef<int64_t>(node.inputs).take_front(2)) {
      const TensorRecord &input = graph.tensors[inputIndex];
      if (input.type.getNumElements() > 131072)
        return false;
      SmallVector<int64_t, 4> inputShape = broadcastTensor4d(input.type);
      for (auto [inputDimension, outputDimension] :
           llvm::zip_equal(inputShape, output))
        if (inputDimension != 1 && inputDimension != outputDimension)
          return false;
    }
    return llvm::all_of(output, [](int64_t dimension) {
      return dimension > 0 && dimension <= 65535;
    });
  };
  auto supportsLargeActivation = [&]() {
    if (!hasF32InputsAndOutput(1))
      return false;
    const TensorRecord &input = graph.tensors[node.inputs[0]];
    const TensorRecord &output = graph.tensors[node.output];
    return input.type.getNumElements() == output.type.getNumElements() &&
           output.type.getNumElements() > 0 &&
           output.type.getNumElements() <= 32768;
  };
  if (isMetadataOnlyAlias(node))
    return isDeviceAlias(graph, node);
  if (node.operation == "Conv")
    return node.provider == "udondnnConvolutionForward" &&
           hasF32InputsAndOutput(2);
  if (node.operation == "Add")
    return node.provider == "udonatenAdd" && supportsLargePointwise();
  if (node.operation == "Mul")
    return node.provider == "udonatenMul" && supportsLargePointwise();
  if (node.operation == "Relu")
    return node.provider == "udonatenRelu" && supportsLargeActivation();
  if (node.operation == "Sigmoid")
    return node.provider == "udonatenSigmoid" && supportsLargeActivation();
  if (node.operation == "Tanh")
    return node.provider == "udonatenTanh" && supportsLargeActivation();
  if (node.operation == "AveragePool")
    return node.provider == "udondnnPoolingForward" && hasF32InputsAndOutput(1);
  if (node.operation == "Cos" || node.operation == "Sin" ||
      node.operation == "Neg")
    return node.provider == "udononnx" + node.operation &&
           supportsLargeActivation() &&
           graph.tensors[node.output].type.getNumElements() <= 16384;
  if (node.operation == "MatMulNBits")
    return isDeviceQ4Node(graph, node);
  if (node.operation == "RmsNorm")
    return isDeviceRmsNormNode(graph, node);
  if (node.operation == "Softmax")
    return isDeviceSoftmaxNode(graph, node);
  if (node.operation == "Transpose")
    return deviceTransposePermutation(graph, node).has_value();
  if (node.operation == "Slice")
    return deviceSliceShape(graph, node).has_value();
  if (node.operation == "Expand")
    return isDeviceExpandNode(graph, node);
  if (node.operation == "Concat")
    return deviceConcatShape(graph, node).has_value();
  if (node.operation == "Gather")
    return deviceGatherShape(graph, node).has_value();
  if (node.operation == "Cast")
    return isDeviceCastI64F32Node(graph, node);
  if (node.operation == "Equal")
    return isDeviceEqualNode(graph, node);
  if (node.operation == "Where")
    return isDeviceWhereNode(graph, node);
  if (node.operation == "ScatterND")
    return deviceScatterNDShape(graph, node).has_value();
  if (node.operation == "MatMul")
    return deviceMatMulShape(graph, node).has_value();
  if (node.operation == "Gemm") {
    if (node.provider != "udonblasSgemm" || !hasF32InputsAndOutput(2))
      return false;
    const TensorRecord &left = graph.tensors[node.inputs[0]];
    const TensorRecord &right = graph.tensors[node.inputs[1]];
    const TensorRecord &output = graph.tensors[node.output];
    // The arena SGEMM kernels bound K, the operand capacities and the output
    // surface; anything larger must not reach them.
    return left.type.getRank() == 2 && right.type.getRank() == 2 &&
           left.type.getDimSize(1) >= 1 &&
           left.type.getDimSize(1) <= deviceSgemmMaxK &&
           left.type.getNumElements() <= 131072 &&
           right.type.getNumElements() <= 131072 &&
           output.type.getNumElements() <= 16384;
  }
  return false;
}

bool supportsDeviceGraph(const GraphRecord &graph) {
  return llvm::all_of(graph.nodes, [&](const NodeRecord &node) {
    return supportsDeviceNode(graph, node);
  });
}
SmallVector<unsigned> deviceStorage(const GraphRecord &graph) {
  SmallVector<unsigned> storage(graph.tensors.size());
  std::iota(storage.begin(), storage.end(), 0u);
  for (const NodeRecord &node : graph.nodes)
    if (isDeviceAlias(graph, node))
      storage[node.output] = storage[static_cast<unsigned>(node.inputs[0])];

  // Assign non-overlapping intermediate lifetimes to the same physical
  // allocation. A dispatch may read every input while writing its output, so
  // a slot becomes reusable only after the last input use, never at that same
  // node. Graph outputs remain live through the final public-buffer copy.
  SmallVector<int64_t> lastUse(graph.tensors.size(), -1);
  for (auto [nodeIndex, node] : llvm::enumerate(graph.nodes))
    for (int64_t input : node.inputs)
      if (input >= 0) {
        unsigned root = storage[static_cast<unsigned>(input)];
        lastUse[root] =
            std::max(lastUse[root], static_cast<int64_t>(nodeIndex));
      }
  for (unsigned output : graph.outputs)
    lastUse[storage[output]] = static_cast<int64_t>(graph.nodes.size());

  struct Slot {
    unsigned root;
    int64_t bytes;
    int64_t availableAfter;
  };
  SmallVector<Slot> slots;
  for (auto [nodeIndex, node] : llvm::enumerate(graph.nodes)) {
    if (isDeviceAlias(graph, node))
      continue;
    unsigned output = node.output;
    const TensorRecord &tensor = graph.tensors[output];
    if (tensor.staticResourceId >= 0 ||
        llvm::is_contained(graph.inputs, output) ||
        llvm::is_contained(graph.outputs, output))
      continue;
    int64_t bytes = deviceAllocationBytes(graph, output);
    Slot *best = nullptr;
    for (Slot &slot : slots)
      if (slot.availableAfter < static_cast<int64_t>(nodeIndex) &&
          slot.bytes >= bytes && (!best || slot.bytes < best->bytes))
        best = &slot;
    if (best) {
      storage[output] = best->root;
      best->availableAfter = lastUse[output];
    } else {
      slots.push_back({output, bytes, lastUse[output]});
    }
  }
  for (unsigned &root : storage)
    while (storage[root] != root)
      root = storage[root];
  return storage;
}

int64_t bindingCapacityBytes(const GraphRecord &graph, unsigned tensorIndex) {
  const TensorRecord &tensor = graph.tensors[tensorIndex];
  return std::max<int64_t>(planElementByteSize(tensor.type.getElementType()),
                           deviceAllocationBytes(graph, tensorIndex));
}

SmallVector<int64_t> mutableBufferBytes(const GraphRecord &graph,
                                        bool deviceGraph) {
  SmallVector<int64_t> bytes;
  bytes.reserve(graph.inputs.size() + graph.outputs.size());
  for (unsigned tensorIndex : graph.inputs)
    bytes.push_back(bindingCapacityBytes(graph, tensorIndex));
  for (unsigned tensorIndex : graph.outputs)
    bytes.push_back(bindingCapacityBytes(graph, tensorIndex));
  if (!deviceGraph)
    return bytes;

  SmallVector<unsigned> storage = deviceStorage(graph);
  for (auto [tensorIndex, tensor] : llvm::enumerate(graph.tensors)) {
    if (storage[tensorIndex] != tensorIndex || tensor.staticResourceId >= 0 ||
        llvm::is_contained(graph.inputs, tensorIndex) ||
        (llvm::is_contained(graph.outputs, tensorIndex) &&
         !needsQ4Surface(graph, tensorIndex)))
      continue;
    bytes.push_back(deviceAllocationBytes(graph, tensorIndex));
  }
  if (int64_t scratchBytes = operatorScratchBytes(graph))
    bytes.push_back(scratchBytes);
  return bytes;
}

SmallVector<unsigned> ownedDeviceBuffers(const GraphRecord &graph,
                                         bool deviceGraph,
                                         ArrayRef<unsigned> storage) {
  SmallVector<unsigned> owned;
  if (!deviceGraph)
    return owned;
  for (auto [tensorIndex, tensor] : llvm::enumerate(graph.tensors))
    if (storage[tensorIndex] == tensorIndex && tensor.staticResourceId < 0 &&
        !llvm::is_contained(graph.inputs, tensorIndex) &&
        (!llvm::is_contained(graph.outputs, tensorIndex) ||
         needsQ4Surface(graph, tensorIndex)))
      owned.push_back(tensorIndex);
  return owned;
}

FailureOr<HostPlanRecord> readHostPlan(ModuleOp module,
                                       const GraphRecord &graph) {
  auto plan = module->getAttrOfType<DictionaryAttr>(hostGraphPlanAttr);
  auto deviceGraph = plan ? plan.getAs<BoolAttr>("device_graph") : BoolAttr();
  auto storage = plan ? plan.getAs<DenseI64ArrayAttr>("device_storage")
                      : DenseI64ArrayAttr();
  auto owned = plan ? plan.getAs<DenseI64ArrayAttr>("owned_device_buffers")
                    : DenseI64ArrayAttr();
  auto scratch =
      plan ? plan.getAs<IntegerAttr>("operator_scratch_bytes") : IntegerAttr();
  auto bufferBytes = plan
                         ? plan.getAs<DenseI64ArrayAttr>("mutable_buffer_bytes")
                         : DenseI64ArrayAttr();
  auto handleCapacity =
      plan ? plan.getAs<IntegerAttr>("mutable_handle_capacity") : IntegerAttr();
  if (!deviceGraph || !storage || !owned || !scratch || !bufferBytes ||
      !handleCapacity || storage.size() != graph.tensors.size() ||
      scratch.getInt() < 0 || handleCapacity.getInt() <= 0 ||
      static_cast<int64_t>(bufferBytes.size()) != handleCapacity.getInt())
    return module.emitError("ONNX Host graph plan is malformed");
  HostPlanRecord result{deviceGraph.getValue(), {}, {},
                        scratch.getInt(),       {}, handleCapacity.getInt()};
  result.deviceStorage.reserve(storage.size());
  for (int64_t value : storage.asArrayRef()) {
    if (value < 0 || value >= static_cast<int64_t>(graph.tensors.size()))
      return module.emitError("ONNX Host graph plan has invalid storage");
    result.deviceStorage.push_back(static_cast<unsigned>(value));
  }
  llvm::SmallDenseSet<unsigned> seenOwned;
  for (int64_t value : owned.asArrayRef()) {
    if (!result.deviceGraph || value < 0 ||
        value >= static_cast<int64_t>(graph.tensors.size()) ||
        result.deviceStorage[value] != static_cast<unsigned>(value) ||
        !seenOwned.insert(static_cast<unsigned>(value)).second)
      return module.emitError(
          "ONNX Host graph plan has invalid owned device buffers");
    result.ownedDeviceBuffers.push_back(static_cast<unsigned>(value));
  }
  for (int64_t value : bufferBytes.asArrayRef())
    result.mutableBufferBytes.push_back(value);
  return result;
}
} // namespace vrc
