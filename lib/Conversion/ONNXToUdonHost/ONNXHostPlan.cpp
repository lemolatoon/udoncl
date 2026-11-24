#include "ONNXHostPlan.h"

#include "ONNXHostFusion.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"

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
         node.operation == "Unsqueeze" || node.operation == "BoundedPrefix";
}

/// Elements of `shape` after (inner) and before (outer) `axis`.
static std::pair<int64_t, int64_t> aroundAxis(ArrayRef<int64_t> shape,
                                              int64_t axis) {
  int64_t outer = 1, inner = 1;
  for (auto [index, extent] : llvm::enumerate(shape)) {
    if (static_cast<int64_t>(index) < axis)
      outer *= extent;
    else if (static_cast<int64_t>(index) > axis)
      inner *= extent;
  }
  return {outer, inner};
}

int64_t dimensionCapacity(const GraphRecord &graph, StringRef symbol) {
  int64_t capacity = 0;
  for (ArrayRef<unsigned> tensors :
       {ArrayRef<unsigned>(graph.inputs), ArrayRef<unsigned>(graph.outputs)})
    for (unsigned index : tensors) {
      const TensorRecord &tensor = graph.tensors[index];
      if (tensor.dynamic && tensor.dynamic->symbol == symbol)
        capacity =
            std::max(capacity, tensor.type.getDimSize(tensor.dynamic->axis));
    }
  return capacity;
}

std::optional<DynamicLayout> dynamicLayout(const GraphRecord &graph,
                                           unsigned tensor) {
  if (!graph.tensors[tensor].dynamic)
    return std::nullopt;
  for (bool changed = true; changed;) {
    changed = false;
    for (const NodeRecord &node : graph.nodes)
      if (node.output == tensor && isMetadataOnlyAlias(node) &&
          !node.inputs.empty() && node.inputs[0] >= 0) {
        tensor = static_cast<unsigned>(node.inputs[0]);
        changed = true;
        break;
      }
  }
  if (const unsigned *input = llvm::find(graph.inputs, tensor);
      input != graph.inputs.end())
    return DynamicLayout{DynamicLayout::Binding, false,
                         static_cast<unsigned>(input - graph.inputs.begin())};
  const TensorRecord &owner = graph.tensors[tensor];
  if (owner.staticResourceId >= 0 || owner.initialValue)
    return DynamicLayout{DynamicLayout::Bound};
  if (const unsigned *output = llvm::find(graph.outputs, tensor);
      output != graph.outputs.end())
    return DynamicLayout{DynamicLayout::Binding, true,
                         static_cast<unsigned>(output - graph.outputs.begin())};
  return DynamicLayout{DynamicLayout::Dense};
}

bool hasDenseDynamicLayout(const GraphRecord &graph, unsigned tensor) {
  const TensorRecord &record = graph.tensors[tensor];
  if (!record.dynamic)
    return true;
  if (dynamicLayout(graph, tensor)->kind == DynamicLayout::Dense)
    return true;
  return aroundAxis(record.type.getShape(), record.dynamic->axis).first == 1;
}

unsigned dimensionIndex(const GraphRecord &graph, StringRef symbol) {
  for (auto [index, dimension] : llvm::enumerate(graph.dimensionBounds))
    if (dimension.first == symbol)
      return index;
  llvm_unreachable("unknown bounded dimension");
}

/// Whether a MatMul with tensors of a bounded dimension runs on the arena
/// SGEMM with run-time m, n and k: the batch axes are static, an operand
/// whose dynamic axis is its inner (column) axis is dense, and the result,
/// which the kernel writes dense, is dense.
static bool supportsDynamicMatMul(const GraphRecord &graph,
                                  const NodeRecord &node) {
  if (!deviceMatMulShape(graph, node))
    return false;
  for (int64_t tensor :
       {node.inputs[0], node.inputs[1], static_cast<int64_t>(node.output)}) {
    const TensorRecord &record = graph.tensors[tensor];
    if (!record.dynamic)
      continue;
    int64_t rank = record.type.getRank();
    if (record.dynamic->axis < rank - 2)
      return false;
    bool inner = record.dynamic->axis == rank - 1;
    if ((inner || tensor == node.output) &&
        !hasDenseDynamicLayout(graph, static_cast<unsigned>(tensor)))
      return false;
  }
  return true;
}

std::optional<std::string> unsupportedDynamicExtent(const GraphRecord &graph,
                                                    const NodeRecord &node) {
  const TensorRecord &output = graph.tensors[node.output];
  const TensorRecord *dynamic = output.dynamic ? &output : nullptr;
  for (int64_t input : node.inputs)
    if (input >= 0 && !dynamic && graph.tensors[input].dynamic)
      dynamic = &graph.tensors[input];
  if (!dynamic)
    return std::nullopt;
  auto diagnostic = [&](const TensorRecord &tensor) {
    return ("ONNX operator " + node.operation +
            " does not support a dynamic extent on axis " +
            Twine(tensor.dynamic->axis) + " (dimension " +
            tensor.dynamic->symbol + ")")
        .str();
  };
  // A view keeps the run-time layout when the dynamic extent stays the same
  // and keeps the elements on either side of it: it then reads the same
  // words at every extent. A bounded prefix is the leading block of its
  // constant at the bound.
  if (node.operation == "MatMul" && supportsDynamicMatMul(graph, node))
    return std::nullopt;
  if (isMetadataOnlyAlias(node) && !node.inputs.empty() &&
      node.inputs[0] >= 0) {
    const TensorRecord &input = graph.tensors[node.inputs[0]];
    if (node.operation == "BoundedPrefix")
      return input.dynamic || !output.dynamic
                 ? std::optional(diagnostic(*dynamic))
                 : std::nullopt;
    if (input.dynamic && output.dynamic &&
        input.dynamic->symbol == output.dynamic->symbol &&
        input.dynamic->offset == output.dynamic->offset &&
        aroundAxis(input.type.getShape(), input.dynamic->axis) ==
            aroundAxis(output.type.getShape(), output.dynamic->axis))
      return std::nullopt;
  }
  return diagnostic(*dynamic);
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

namespace {
/// Elements a kernel of `limits` reads or writes per operand when its library
/// kernel holds `library`.
int64_t elementLimit(DeviceKernelLimits limits, int64_t library) {
  return limits == DeviceKernelLimits::Library ? library
                                               : deviceSynthesizedSurfaceWords;
}
} // namespace

std::optional<DeviceSliceShape> deviceSliceShape(const GraphRecord &graph,
                                                 const NodeRecord &node,
                                                 DeviceKernelLimits limits) {
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
      input.type.getNumElements() > elementLimit(limits, 16384) ||
      output.type.getNumElements() < 1 ||
      output.type.getNumElements() > elementLimit(limits, 16384))
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

bool isDeviceExpandNode(const GraphRecord &graph, const NodeRecord &node,
                        DeviceKernelLimits limits) {
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
      input.type.getNumElements() < 1 ||
      input.type.getNumElements() > elementLimit(limits, 16384) ||
      output.type.getNumElements() < 1 ||
      output.type.getNumElements() > elementLimit(limits, 131072))
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

namespace {
/// The two-input Concat contract with the element limit of the kernel that
/// runs it: the 16384 words udononnx_concat2_f32 addresses, or the surface of
/// a synthesized gather kernel.
std::optional<DeviceConcatShape> concatShapeWithin(const GraphRecord &graph,
                                                   const NodeRecord &node,
                                                   int64_t maximumElements) {
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
      left.type.getNumElements() > maximumElements ||
      right.type.getNumElements() < 1 ||
      right.type.getNumElements() > maximumElements ||
      output.type.getNumElements() < 1 ||
      output.type.getNumElements() > maximumElements)
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
} // namespace

std::optional<DeviceConcatShape> deviceConcatShape(const GraphRecord &graph,
                                                   const NodeRecord &node,
                                                   DeviceKernelLimits limits) {
  return concatShapeWithin(graph, node, elementLimit(limits, 16384));
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

bool isDeviceSoftmaxNode(const GraphRecord &graph, const NodeRecord &node,
                         DeviceKernelLimits limits) {
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
         input.type.getNumElements() <= elementLimit(limits, 16384) &&
         input.type.getDimSize(axis) >= 1 &&
         input.type.getDimSize(axis) <= (limits == DeviceKernelLimits::Library
                                             ? 512
                                             : deviceSynthesizedMaxReduction);
}

std::optional<SmallVector<int64_t>>
deviceTransposePermutation(const GraphRecord &graph, const NodeRecord &node,
                           DeviceKernelLimits limits) {
  if (node.operation != "Transpose" || node.provider != "udononnxTranspose" ||
      node.inputs.size() != 1 || node.inputs[0] < 0)
    return std::nullopt;
  const TensorRecord &input = graph.tensors[node.inputs[0]];
  const TensorRecord &output = graph.tensors[node.output];
  int64_t rank = input.type.getRank();
  if (!input.type.getElementType().isF32() || rank < 1 || rank > 5 ||
      output.type.getRank() != rank || input.type.getNumElements() < 1 ||
      input.type.getNumElements() > elementLimit(limits, 131072) ||
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
      leftTensor.type.getNumElements() > deviceLargeSgemmOperandWords ||
      rightTensor.type.getNumElements() > deviceLargeSgemmOperandWords ||
      outputTensor.type.getNumElements() > deviceLargeSgemmOutputWords)
    return std::nullopt;
  int64_t leftBatches = leftTensor.type.getNumElements() / (m * k);
  int64_t rightBatches = rightTensor.type.getNumElements() / (k * n);
  int64_t outputBatches = outputTensor.type.getNumElements() / (m * n);
  if (outputBatches < 1 || (leftBatches != 1 && leftBatches != outputBatches) ||
      (rightBatches != 1 && rightBatches != outputBatches))
    return std::nullopt;
  return DeviceMatMulShape{m, n, k, leftBatches, rightBatches, outputBatches};
}

bool needsLargeBatchedSgemm(const DeviceMatMulShape &shape) {
  // The words each operand spans over all batches; a broadcast operand
  // (one batch) spans one matrix.
  int64_t left = shape.leftBatches * shape.m * shape.k;
  int64_t right = shape.rightBatches * shape.k * shape.n;
  return left > deviceBatchedSgemmOperandWords ||
         right > deviceBatchedSgemmOperandWords ||
         shape.outputBatches * shape.m * shape.n >
             deviceBatchedSgemmOutputWords;
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

// Split-K cost model. A Blit of a GEMV-shaped kernel is latency-bound when it
// launches few fragments: each fragment's reduction is one dependent chain
// that the GPU cannot hide behind other fragments. The model estimates one
// dispatch, in GPU cycles, as
//   dispatch overhead + max(throughput, chain * waves)
// where throughput is the dispatch's total statements and Loads over the
// GPU's ALU lanes and texture units, chain the statement and Load latency of
// the longest fragment, and waves the number of times the launched fragments
// fill the GPU. It is the latency/throughput model of
// docs/investigations/2026-11-21-tinyllama-decode-gpu-bound.md (a current
// desktop GPU: 60 SMs, 240 texture units, latencies calibrated to the gate13
// Unity trace). The dispatch overhead is the cost of one more Blit: about
// 10 us of GPU and about 24 us of Host issue (the Udon emulator counts about
// 150 more Udon instructions per added sum dispatch, at the gate13 Host rate
// of about 160 ns per instruction), so a split must pay for its sum dispatch.
// The inputs are only the output count, the reduction length and the
// kernel's per-unit statements and Loads; nothing depends on a model.
namespace {
/// Fragments the GPU keeps in flight.
constexpr int64_t splitKResidentFragments = 65536;
/// Statements and texel Loads the GPU issues per cycle (60 SMs x 128 lanes
/// at two cycles per statement, and 240 texture units).
constexpr int64_t splitKStatementLanes = 3840;
constexpr int64_t splitKTextureUnits = 240;
/// Dependent latency of one statement and one texel Load, in cycles.
constexpr int64_t splitKStatementCycles = 24;
constexpr int64_t splitKLoadCycles = 120;
/// Cycles one more dispatch costs (GPU Blit and Host issue, about 34 us).
constexpr int64_t splitKDispatchCycles = 90000;
/// Statements of a fragment besides its reduction (indexing and the output
/// write), and statements per partial the ordered sum adds.
constexpr int64_t splitKFragmentStatements = 20;
constexpr int64_t splitKSumStatements = 6;

int64_t ceilDivide(int64_t value, int64_t divisor) {
  return (value + divisor - 1) / divisor;
}

/// One dispatch of `fragments` fragments whose longest one executes
/// `chainStatements` statements and `chainLoads` Loads.
int64_t splitKPassCycles(int64_t fragments, int64_t statements, int64_t loads,
                         int64_t chainStatements, int64_t chainLoads) {
  int64_t waves = ceilDivide(fragments, splitKResidentFragments);
  int64_t throughput = ceilDivide(statements, splitKStatementLanes) +
                       ceilDivide(loads, splitKTextureUnits);
  int64_t chain =
      chainStatements * splitKStatementCycles + chainLoads * splitKLoadCycles;
  return splitKDispatchCycles + std::max(throughput, chain * waves);
}
} // namespace

std::optional<DeviceSplitK>
chooseDeviceSplitK(const DeviceSplitKProblem &problem) {
  if (problem.outputs < 1 || problem.units < 1 || problem.loadsPerUnit < 0 ||
      problem.statementsPerUnit < 0 || problem.maxUnitsPerSplit < 1 ||
      problem.maxSplits < 1)
    return std::nullopt;
  int64_t outputs = problem.outputs;
  int64_t statements = outputs * (problem.units * problem.statementsPerUnit +
                                  splitKFragmentStatements);
  int64_t loads = outputs * problem.units * problem.loadsPerUnit;
  std::optional<DeviceSplitK> best;
  int64_t bestCycles = 0;
  for (int64_t candidate = 1;
       candidate <= std::min(problem.maxSplits, problem.units); ++candidate) {
    // Only the smallest split count for each slice length: every split is
    // non-empty and the last one is the shortest.
    int64_t unitsPerSplit = ceilDivide(problem.units, candidate);
    int64_t splits = ceilDivide(problem.units, unitsPerSplit);
    if (splits != candidate || unitsPerSplit > problem.maxUnitsPerSplit ||
        (splits > 1 && splits * outputs > problem.partialWords))
      continue;
    int64_t cycles;
    if (splits == 1) {
      cycles = splitKPassCycles(outputs, statements, loads,
                                problem.units * problem.statementsPerUnit +
                                    splitKFragmentStatements,
                                problem.units * problem.loadsPerUnit);
    } else {
      // The partial pass writes every partial once; the sum pass loads them
      // in ascending split order.
      int64_t partials = splits * outputs;
      cycles = splitKPassCycles(partials,
                                statements + (partials - outputs) *
                                                 splitKFragmentStatements,
                                loads,
                                unitsPerSplit * problem.statementsPerUnit +
                                    splitKFragmentStatements,
                                unitsPerSplit * problem.loadsPerUnit) +
               splitKPassCycles(outputs,
                                partials * splitKSumStatements +
                                    outputs * splitKFragmentStatements,
                                partials,
                                splits * splitKSumStatements +
                                    splitKFragmentStatements,
                                splits);
    }
    // Strictly faster only: a tie keeps fewer splits.
    if (!best || cycles < bestCycles) {
      best = DeviceSplitK{splits, unitsPerSplit};
      bestCycles = cycles;
    }
  }
  return best;
}

std::optional<DeviceSplitK> deviceQ4SplitK(const GraphRecord &graph,
                                           const NodeRecord &node) {
  if (!isDeviceQ4Node(graph, node))
    return std::nullopt;
  int64_t k = integerAttribute(node, "K", 0);
  int64_t n = integerAttribute(node, "N", 0);
  int64_t m = graph.tensors[node.inputs[0]].type.getNumElements() / k;
  // One 32-element block of udononnx_matmul_nbits_q4 executes 37 Loads (1
  // scale, 4 packed words, 32 activations) and about 583 HLSL statements
  // (counted on the generated kernel).
  return chooseDeviceSplitK(
      DeviceSplitKProblem{m * n, (k + 31) / 32, 37, 583, deviceQ4DispatchBlocks,
                          deviceQ4MaxSplits, q4SurfaceWords});
}

// The row-reduction schedule uses the split-K model above. Each row element
// is one evaluation of the row operand; the one-pass kernel evaluates the
// whole row twice in every fragment (maximum, then the sum of exponentials)
// and its own element once. The two-pass schedule runs
//  1. a statistics pass of 2 x rows x chunks fragments: the first plane holds
//     the maximum of each chunk, the second the sum of exp(sample - chunk
//     maximum), which re-derives the chunk maximum first, and
//  2. the elementwise pass, whose fragments combine the chunk partials of
//     their row (three Loads per chunk) and evaluate their own element.
namespace {
/// Statements of a row element besides its operand: the compare and select
/// of the maximum, and the subtraction, exponential and addition of the sum.
constexpr int64_t rowMaximumStatements = 2;
constexpr int64_t rowSumStatements = 4;
/// Statements and Loads with which the elementwise pass combines one chunk.
constexpr int64_t rowCombineStatements = 44;
constexpr int64_t rowCombineLoads = 3;
} // namespace

DeviceRowReduction
chooseDeviceRowReduction(const DeviceRowReductionProblem &problem) {
  DeviceRowReduction onePass{0, problem.extent};
  if (problem.rows < 1 || problem.extent < 2 || problem.loadsPerElement < 0 ||
      problem.statementsPerElement < 0)
    return onePass;
  int64_t rows = problem.rows, extent = problem.extent;
  int64_t loads = problem.loadsPerElement;
  int64_t maximum = problem.statementsPerElement + rowMaximumStatements;
  int64_t sum = problem.statementsPerElement + rowSumStatements;
  int64_t outputs = rows * extent;
  int64_t onePassChain = extent * (maximum + sum) +
                         problem.statementsPerElement +
                         splitKFragmentStatements;
  int64_t bestCycles = splitKPassCycles(outputs, outputs * onePassChain,
                                        outputs * (2 * extent + 1) * loads,
                                        onePassChain, (2 * extent + 1) * loads);
  DeviceRowReduction best = onePass;
  for (int64_t candidate = 2;
       candidate <= std::min(extent, deviceRowReductionMaxChunks);
       ++candidate) {
    int64_t chunkLength = ceilDivide(extent, candidate);
    int64_t chunks = ceilDivide(extent, chunkLength);
    if (chunks != candidate || 2 * rows * chunks > problem.statisticsWords)
      continue;
    int64_t partials = rows * chunks;
    int64_t sumChain = chunkLength * (maximum + sum) + splitKFragmentStatements;
    int64_t statistics = splitKPassCycles(
        2 * partials,
        outputs * (2 * maximum + sum) + 2 * partials * splitKFragmentStatements,
        3 * outputs * loads, sumChain, 2 * chunkLength * loads);
    int64_t combineChain = chunks * rowCombineStatements +
                           problem.statementsPerElement + rowSumStatements +
                           splitKFragmentStatements;
    int64_t combineLoads = chunks * rowCombineLoads + loads;
    int64_t elementwise =
        splitKPassCycles(outputs, outputs * combineChain,
                         outputs * combineLoads, combineChain, combineLoads);
    // The second pass must save more than a tenth of the one-pass time, about
    // the resolution of the model; among two-pass schedules a tie keeps fewer
    // chunks.
    int64_t cycles = statistics + elementwise;
    if (best.chunks ? cycles < bestCycles : cycles * 10 < bestCycles * 9) {
      best = DeviceRowReduction{chunks, chunkLength};
      bestCycles = statistics + elementwise;
    }
  }
  return best;
}

std::optional<DeviceSoftmaxRows> deviceSoftmaxRows(const GraphRecord &graph,
                                                   const NodeRecord &node) {
  if (!isDeviceSoftmaxNode(graph, node, DeviceKernelLimits::Synthesized))
    return std::nullopt;
  RankedTensorType type = graph.tensors[node.inputs[0]].type;
  int64_t axis = integerAttribute(node, "axis", -1);
  if (axis < 0)
    axis += type.getRank();
  DeviceSoftmaxRows rows{1, type.getDimSize(axis), 1};
  for (int64_t extent : type.getShape().take_front(axis))
    rows.outer *= extent;
  for (int64_t extent : type.getShape().drop_front(axis + 1))
    rows.inner *= extent;
  return rows;
}

DeviceRowReduction deviceSoftmaxSchedule(const GraphRecord &graph,
                                         const NodeRecord &node) {
  std::optional<DeviceSoftmaxRows> rows = deviceSoftmaxRows(graph, node);
  if (!rows)
    return DeviceRowReduction{0, 0};
  // A fused Softmax evaluates an operand of at most
  // deviceFusionMaxReductionOperandLoads Loads per element; one Load
  // decodes a texel in about 12 statements, and indexing adds about 3.
  int64_t loads = deviceFusionMaxReductionOperandLoads;
  return chooseDeviceRowReduction(
      DeviceRowReductionProblem{rows->rows(), rows->extent, loads,
                                12 * loads + 3, deviceRowStatisticsWords});
}

static bool hasTwoPassSoftmax(const GraphRecord &graph) {
  return llvm::any_of(graph.nodes, [&](const NodeRecord &node) {
    return deviceSoftmaxSchedule(graph, node).chunks > 0;
  });
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
  if (hasTwoPassSoftmax(graph))
    bytes = std::max(bytes, deviceRowStatisticsWords * 4);
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
  for (const SequenceStateRecord &state : graph.sequenceStates)
    if (graph.inputs[state.inputIndex] == tensorIndex)
      bytes = std::max(bytes, deviceAllocationBytes(
                                  graph, graph.outputs[state.outputIndex]));
  return needsQ4Surface(graph, tensorIndex) ? std::max(bytes, q4SurfaceBytes)
                                            : bytes;
}

SmallVector<unsigned> sequenceStateBanks(const GraphRecord &graph) {
  SmallVector<unsigned> banks;
  if (graph.sequenceStates.empty() || !supportsDeviceGraph(graph))
    return banks;
  for (auto [stateIndex, state] : llvm::enumerate(graph.sequenceStates)) {
    unsigned past = graph.inputs[state.inputIndex];
    unsigned present = graph.outputs[state.outputIndex];
    if (llvm::count(graph.inputs, past) != 1 ||
        llvm::count(graph.outputs, present) != 1 ||
        llvm::is_contained(graph.outputs, past))
      continue;
    const NodeRecord *concat = nullptr;
    bool otherReader = false;
    for (const NodeRecord &node : graph.nodes) {
      if (node.output == present)
        concat = &node;
      for (auto [operand, input] : llvm::enumerate(node.inputs))
        if (input == static_cast<int64_t>(past) &&
            (node.output != present || operand != 0))
          otherReader = true;
    }
    std::optional<DeviceFusionKind> kind =
        concat ? deviceFusionKind(graph, *concat) : std::nullopt;
    if (otherReader || !kind || *kind != DeviceFusionKind::Concat ||
        concat->inputs.size() != 2 ||
        concat->inputs[0] != static_cast<int64_t>(past))
      continue;
    banks.push_back(static_cast<unsigned>(stateIndex));
  }
  return banks;
}

namespace {
/// NumPy broadcasting of `input` to `output` on the real shapes, as a
/// synthesized kernel evaluates it.
bool broadcastsTo(RankedTensorType input, RankedTensorType output) {
  int64_t offset = output.getRank() - input.getRank();
  if (offset < 0)
    return false;
  for (int64_t axis = 0; axis < input.getRank(); ++axis) {
    int64_t extent = input.getDimSize(axis);
    if (extent != 1 && extent != output.getDimSize(axis + offset))
      return false;
  }
  return true;
}

/// supportsDeviceNode within the element limits of `limits`. Only the
/// per-element, re-indexing and Softmax nodes a synthesized kernel can
/// evaluate have synthesized limits; every other node needs its library
/// kernel.
bool supportsDeviceNodeWithin(const GraphRecord &graph, const NodeRecord &node,
                              DeviceKernelLimits limits) {
  bool synthesized = limits == DeviceKernelLimits::Synthesized;
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
    if (synthesized) {
      RankedTensorType output = graph.tensors[node.output].type;
      return output.getNumElements() >= 1 &&
             output.getNumElements() <= deviceSynthesizedSurfaceWords &&
             llvm::all_of(ArrayRef<int64_t>(node.inputs).take_front(2),
                          [&](int64_t input) {
                            RankedTensorType type = graph.tensors[input].type;
                            return type.hasStaticShape() &&
                                   type.getNumElements() >= 1 &&
                                   broadcastsTo(type, output);
                          });
    }
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
  auto supportsLargeActivation = [&](int64_t libraryWords) {
    if (!hasF32InputsAndOutput(1))
      return false;
    const TensorRecord &input = graph.tensors[node.inputs[0]];
    const TensorRecord &output = graph.tensors[node.output];
    return input.type.getNumElements() == output.type.getNumElements() &&
           output.type.getNumElements() > 0 &&
           output.type.getNumElements() <= elementLimit(limits, libraryWords);
  };
  if (isMetadataOnlyAlias(node))
    return !synthesized && isDeviceAlias(graph, node);
  if (node.operation == "Conv")
    return !synthesized && node.provider == "udondnnConvolutionForward" &&
           hasF32InputsAndOutput(2);
  if (node.operation == "Add")
    return node.provider == "udonatenAdd" && supportsLargePointwise();
  if (node.operation == "Mul")
    return node.provider == "udonatenMul" && supportsLargePointwise();
  if (node.operation == "Relu")
    return node.provider == "udonatenRelu" && supportsLargeActivation(32768);
  if (node.operation == "Sigmoid")
    return node.provider == "udonatenSigmoid" && supportsLargeActivation(32768);
  if (node.operation == "Tanh")
    return node.provider == "udonatenTanh" && supportsLargeActivation(32768);
  if (node.operation == "AveragePool")
    return !synthesized && node.provider == "udondnnPoolingForward" &&
           hasF32InputsAndOutput(1);
  if (node.operation == "Cos" || node.operation == "Sin" ||
      node.operation == "Neg")
    return node.provider == "udononnx" + node.operation &&
           supportsLargeActivation(16384);
  if (node.operation == "Softmax")
    return isDeviceSoftmaxNode(graph, node, limits);
  if (node.operation == "Transpose")
    return deviceTransposePermutation(graph, node, limits).has_value();
  if (node.operation == "Slice")
    return deviceSliceShape(graph, node, limits).has_value();
  if (node.operation == "Expand")
    return isDeviceExpandNode(graph, node, limits);
  if (node.operation == "Concat")
    return deviceConcatShape(graph, node, limits).has_value();
  if (synthesized)
    return false;
  if (node.operation == "MatMulNBits")
    return isDeviceQ4Node(graph, node);
  if (node.operation == "RmsNorm")
    return isDeviceRmsNormNode(graph, node);
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
} // namespace

bool supportsDeviceNode(const GraphRecord &graph, const NodeRecord &node) {
  return supportsDeviceNodeWithin(graph, node, DeviceKernelLimits::Library) ||
         isSynthesizedDeviceNode(graph, node);
}

bool isSynthesizedDeviceNode(const GraphRecord &graph, const NodeRecord &node) {
  return !supportsDeviceNodeWithin(graph, node, DeviceKernelLimits::Library) &&
         supportsDeviceNodeWithin(graph, node, DeviceKernelLimits::Synthesized);
}

bool supportsDeviceGraph(const GraphRecord &graph) {
  return llvm::all_of(graph.nodes, [&](const NodeRecord &node) {
    return supportsDeviceNode(graph, node);
  });
}

SmallVector<unsigned> deviceStorage(const GraphRecord &graph,
                                    ArrayRef<unsigned> fusionRoots) {
  SmallVector<unsigned> storage(graph.tensors.size());
  std::iota(storage.begin(), storage.end(), 0u);
  for (const NodeRecord &node : graph.nodes)
    if (isDeviceAlias(graph, node))
      storage[node.output] = storage[static_cast<unsigned>(node.inputs[0])];

  // Assign non-overlapping intermediate lifetimes to the same physical
  // allocation. A dispatch may read every input while writing its output, so
  // a slot becomes reusable only after the last input use, never at that same
  // node. Graph outputs remain live through the final public-buffer copy.
  // An interior node of a fusion group has no dispatch of its own: the
  // dispatch of the group's root reads the node's operands, which therefore
  // stay live until the root. A result only takes a slot whose last reader
  // dispatched earlier, so the root never shares a buffer with such an
  // operand and a fused kernel never reads the buffer it writes.
  SmallVector<int64_t> dispatchIndex(fusionRoots.begin(), fusionRoots.end());
  SmallVector<int64_t> lastUse(graph.tensors.size(), -1);
  for (auto [nodeIndex, node] : llvm::enumerate(graph.nodes))
    for (int64_t input : node.inputs)
      if (input >= 0) {
        unsigned root = storage[static_cast<unsigned>(input)];
        lastUse[root] = std::max(lastUse[root], dispatchIndex[nodeIndex]);
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
    if (dispatchIndex[nodeIndex] != static_cast<int64_t>(nodeIndex)) {
      // Never written or read on the Device; it shares the root's result.
      storage[output] = graph.nodes[dispatchIndex[nodeIndex]].output;
      continue;
    }
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
                                        bool deviceGraph,
                                        ArrayRef<unsigned> storage) {
  SmallVector<int64_t> bytes;
  bytes.reserve(graph.inputs.size() + graph.outputs.size());
  for (unsigned tensorIndex : graph.inputs)
    bytes.push_back(bindingCapacityBytes(graph, tensorIndex));
  for (unsigned tensorIndex : graph.outputs)
    bytes.push_back(bindingCapacityBytes(graph, tensorIndex));
  if (!deviceGraph)
    return bytes;

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

SmallVector<unsigned> resolvedRunTensors(const GraphRecord &graph,
                                         const HostPlanRecord &plan) {
  llvm::SetVector<unsigned> tensors;
  if (!plan.deviceGraph)
    return {};
  tensors.insert(graph.inputs.begin(), graph.inputs.end());
  tensors.insert(graph.outputs.begin(), graph.outputs.end());
  tensors.insert(plan.ownedDeviceBuffers.begin(),
                 plan.ownedDeviceBuffers.end());
  return SmallVector<unsigned>(tensors.begin(), tensors.end());
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

bool isTransitionalSequenceStateHost(ModuleOp module) {
  return module->hasAttr("udoncl.onnx_sequence_states");
}

FailureOr<HostPlanRecord> readHostPlan(ModuleOp module,
                                       const GraphRecord &graph) {
  auto plan = module->getAttrOfType<DictionaryAttr>(hostGraphPlanAttr);
  auto deviceGraph = plan ? plan.getAs<BoolAttr>("device_graph") : BoolAttr();
  auto storage = plan ? plan.getAs<DenseI64ArrayAttr>("device_storage")
                      : DenseI64ArrayAttr();
  auto fusionRoots = plan ? plan.getAs<DenseI64ArrayAttr>("device_fusion_roots")
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
  if (!deviceGraph || !storage || !fusionRoots || !owned || !scratch ||
      !bufferBytes || !handleCapacity ||
      storage.size() != graph.tensors.size() ||
      fusionRoots.size() != graph.nodes.size() || scratch.getInt() < 0 ||
      handleCapacity.getInt() <= 0 ||
      static_cast<int64_t>(bufferBytes.size()) != handleCapacity.getInt())
    return module.emitError("ONNX Host graph plan is malformed");
  HostPlanRecord result{
      deviceGraph.getValue(), {}, {}, {}, scratch.getInt(), {},
      handleCapacity.getInt()};
  result.deviceStorage.reserve(storage.size());
  for (int64_t value : storage.asArrayRef()) {
    if (value < 0 || value >= static_cast<int64_t>(graph.tensors.size()))
      return module.emitError("ONNX Host graph plan has invalid storage");
    result.deviceStorage.push_back(static_cast<unsigned>(value));
  }
  for (auto [nodeIndex, root] : llvm::enumerate(fusionRoots.asArrayRef())) {
    if (root < 0 || root >= static_cast<int64_t>(graph.nodes.size()) ||
        (!result.deviceGraph && root != static_cast<int64_t>(nodeIndex)))
      return module.emitError("ONNX Host graph plan has invalid fusion roots");
    result.deviceFusionRoots.push_back(static_cast<unsigned>(root));
  }
  if (!DeviceFusion::isValid(graph, result.deviceFusionRoots))
    return module.emitError("ONNX Host graph plan has invalid fusion roots");
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
  // Absent in plans that bank no state.
  if (Attribute banksAttr = plan.get("sequence_state_banks")) {
    auto banks = dyn_cast<DenseI64ArrayAttr>(banksAttr);
    if (!banks)
      return module.emitError("ONNX Host graph plan has invalid sequence "
                              "state banks");
    SmallVector<unsigned> bankable = sequenceStateBanks(graph);
    for (int64_t value : banks.asArrayRef()) {
      if (value < 0 || !llvm::is_contained(bankable, value) ||
          llvm::is_contained(result.sequenceStateBanks, value))
        return module.emitError("ONNX Host graph plan has invalid sequence "
                                "state banks");
      result.sequenceStateBanks.push_back(static_cast<unsigned>(value));
    }
  }
  return result;
}
} // namespace vrc
