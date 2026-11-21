#include "ONNXHostSequenceExtents.h"

#include "ONNXHostFusion.h"
#include "ONNXHostPlan.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/bit.h"

#include <cmath>
#include <functional>

using namespace mlir;

namespace vrc {
namespace {
int64_t normalizedAxis(int64_t axis, int64_t rank) {
  return axis < 0 ? axis + rank : axis;
}

int64_t product(ArrayRef<int64_t> extents) {
  int64_t result = 1;
  for (int64_t extent : extents)
    result *= extent;
  return result;
}

/// The axis of `to` that holds the coordinates of axis `axis` of `from` when
/// both describe the same row-major elements: the extents before it multiply
/// to the same number and the axis has the same extent, so an element's
/// coordinate on it is the same in both shapes.
std::optional<int64_t> aliasAxis(ArrayRef<int64_t> from, int64_t axis,
                                 ArrayRef<int64_t> to) {
  int64_t outer = product(from.take_front(axis));
  int64_t before = 1;
  for (auto [index, extent] : llvm::enumerate(to)) {
    if (before == outer && extent == from[axis])
      return static_cast<int64_t>(index);
    before *= extent;
  }
  return std::nullopt;
}

ArrayRef<int64_t> shapeOf(const GraphRecord &graph, int64_t tensor) {
  return graph.tensors[tensor].type.getShape();
}

int64_t carriedRows(const GraphRecord &graph, unsigned stateIndex) {
  const SequenceStateRecord &state = graph.sequenceStates[stateIndex];
  return state.capacity - state.step;
}
} // namespace

bool DeviceSequenceExtents::used() const {
  return llvm::any_of(skippedRows,
                      [](const auto &state) { return state.has_value(); }) ||
         llvm::any_of(constantPrefix,
                      [](const auto &prefix) { return prefix.has_value(); });
}

int64_t sequenceStateZeroRows(const GraphRecord &graph, unsigned stateIndex,
                              int64_t advances) {
  const SequenceStateRecord &state = graph.sequenceStates[stateIndex];
  return std::max<int64_t>(0, carriedRows(graph, stateIndex) -
                                  advances * state.step);
}

int64_t sequenceStateFillAgeLimit(const GraphRecord &graph) {
  int64_t advances = 0;
  for (auto [stateIndex, state] : llvm::enumerate(graph.sequenceStates))
    advances = std::max<int64_t>(
        advances, (carriedRows(graph, stateIndex) + state.step - 1) /
                      std::max<int64_t>(state.step, 1));
  return advances + 1;
}

DeviceSequenceExtents deviceSequenceExtents(const GraphRecord &graph) {
  DeviceSequenceExtents extents;
  extents.zero.resize(graph.tensors.size());
  extents.skippedRows.resize(graph.nodes.size());
  extents.constantPrefix.resize(graph.nodes.size());
  if (graph.sequenceStates.empty() || !supportsDeviceGraph(graph))
    return extents;

  // The node that writes each tensor, and every value reader of a tensor.
  SmallVector<int64_t> producers(graph.tensors.size(), -1);
  SmallVector<SmallVector<std::pair<unsigned, unsigned>>> readers(
      graph.tensors.size());
  for (auto [nodeIndex, node] : llvm::enumerate(graph.nodes)) {
    producers[node.output] = static_cast<int64_t>(nodeIndex);
    for (auto [operand, input] : llvm::enumerate(node.inputs))
      if (input >= 0)
        readers[input].push_back(
            {static_cast<unsigned>(nodeIndex), static_cast<unsigned>(operand)});
  }

  // Zero prefixes, in node order. Every prefix derived from state s spans
  // an axis of at least carried_s elements, so Z_s never exceeds it.
  for (auto [stateIndex, state] : llvm::enumerate(graph.sequenceStates)) {
    unsigned past = graph.inputs[state.inputIndex];
    int64_t axis =
        normalizedAxis(state.axis, graph.tensors[past].type.getRank());
    if (axis < 0 || axis >= graph.tensors[past].type.getRank() ||
        shapeOf(graph, past)[axis] != carriedRows(graph, stateIndex))
      continue;
    extents.zero[past] =
        DeviceConstantPrefix{{static_cast<unsigned>(stateIndex), axis}, 0u};
  }
  auto spans = [&](const DeviceConstantPrefix &zero, unsigned tensor) {
    return shapeOf(graph, tensor)[zero.prefix.axis] >=
           carriedRows(graph, zero.prefix.state);
  };
  for (const NodeRecord &node : graph.nodes) {
    if (node.inputs.empty() || node.inputs[0] < 0 || extents.zero[node.output])
      continue;
    ArrayRef<int64_t> output = shapeOf(graph, node.output);
    std::optional<DeviceConstantPrefix> result;
    if (isDeviceAlias(graph, node)) {
      if (const auto &zero = extents.zero[node.inputs[0]])
        if (std::optional<int64_t> axis = aliasAxis(
                shapeOf(graph, node.inputs[0]), zero->prefix.axis, output))
          result =
              DeviceConstantPrefix{{zero->prefix.state, *axis}, zero->bits};
    } else if (std::optional<DeviceFusionKind> kind =
                   deviceFusionKind(graph, node)) {
      switch (*kind) {
      case DeviceFusionKind::Concat: {
        // The first operand's elements keep their coordinates.
        const auto &zero = extents.zero[node.inputs[0]];
        int64_t axis = normalizedAxis(integerAttribute(node, "axis", 0),
                                      static_cast<int64_t>(output.size()));
        if (zero && zero->prefix.axis == axis)
          result = zero;
        break;
      }
      case DeviceFusionKind::Transpose: {
        const auto &zero = extents.zero[node.inputs[0]];
        std::optional<SmallVector<int64_t>> permutation =
            deviceTransposePermutation(graph, node,
                                       DeviceKernelLimits::Synthesized);
        if (zero && permutation)
          for (auto [outputAxis, inputAxis] : llvm::enumerate(*permutation))
            if (inputAxis == zero->prefix.axis)
              result = DeviceConstantPrefix{
                  {zero->prefix.state, static_cast<int64_t>(outputAxis)},
                  zero->bits};
        break;
      }
      case DeviceFusionKind::Expand: {
        const auto &zero = extents.zero[node.inputs[0]];
        ArrayRef<int64_t> input = shapeOf(graph, node.inputs[0]);
        int64_t axis = zero ? zero->prefix.axis +
                                  static_cast<int64_t>(output.size()) -
                                  static_cast<int64_t>(input.size())
                            : -1;
        if (zero && axis >= 0 && output[axis] == input[zero->prefix.axis])
          result = DeviceConstantPrefix{{zero->prefix.state, axis}, zero->bits};
        break;
      }
      case DeviceFusionKind::Mul: {
        // A finite constant scalar maps a signed zero to a signed zero, the
        // same for every element.
        for (unsigned operand = 0; operand != 2 && !result; ++operand) {
          int64_t other =
              node.inputs.size() == 2 ? node.inputs[1 - operand] : int64_t(-1);
          const auto &zero = node.inputs[operand] >= 0
                                 ? extents.zero[node.inputs[operand]]
                                 : std::nullopt;
          std::optional<float> scale =
              other >= 0 ? scalarF32(graph.tensors[other]) : std::nullopt;
          if (!zero || !scale || !std::isfinite(*scale) ||
              shapeOf(graph, node.inputs[operand]) != output)
            continue;
          float value = llvm::bit_cast<float>(zero->bits) * *scale;
          result = DeviceConstantPrefix{zero->prefix,
                                        llvm::bit_cast<uint32_t>(value)};
        }
        break;
      }
      default:
        break;
      }
    }
    if (result && spans(*result, node.output))
      extents.zero[node.output] = result;
  }

  // The non-alias tensor an alias chain starts from, with the axis of that
  // tensor that holds `axis` of `tensor`.
  auto aliasSource =
      [&](unsigned tensor,
          int64_t axis) -> std::optional<std::pair<unsigned, int64_t>> {
    while (producers[tensor] >= 0 &&
           isDeviceAlias(graph, graph.nodes[producers[tensor]])) {
      unsigned source =
          static_cast<unsigned>(graph.nodes[producers[tensor]].inputs[0]);
      std::optional<int64_t> moved =
          aliasAxis(shapeOf(graph, tensor), axis, shapeOf(graph, source));
      if (!moved)
        return std::nullopt;
      tensor = source;
      axis = *moved;
    }
    return std::make_pair(tensor, axis);
  };

  // Product rows a MatMul skips.
  for (auto [nodeIndex, node] : llvm::enumerate(graph.nodes)) {
    if (!deviceMatMulShape(graph, node))
      continue;
    unsigned left = static_cast<unsigned>(node.inputs[0]);
    unsigned right = static_cast<unsigned>(node.inputs[1]);
    const auto &zero = extents.zero[right];
    int64_t rightRank = graph.tensors[right].type.getRank();
    int64_t leftRank = graph.tensors[left].type.getRank();
    if (!zero || zero->prefix.axis != rightRank - 2 ||
        (zero->bits & 0x7fffffffu) != 0 ||
        shapeOf(graph, right)[rightRank - 2] <=
            carriedRows(graph, zero->prefix.state))
      continue;
    std::optional<std::pair<unsigned, int64_t>> source =
        aliasSource(left, leftRank - 1);
    if (!source || producers[source->first] < 0)
      continue;
    const NodeRecord &softmax = graph.nodes[producers[source->first]];
    if (!isDeviceSoftmaxNode(graph, softmax, DeviceKernelLimits::Synthesized) ||
        normalizedAxis(integerAttribute(softmax, "axis", -1),
                       graph.tensors[softmax.output].type.getRank()) !=
            source->second)
      continue;
    extents.skippedRows[nodeIndex] = zero->prefix.state;
  }

  // A prefix nobody reads: every reader, through metadata aliases, is a
  // MatMul that skips exactly those rows of its left operand.
  std::function<bool(unsigned, int64_t, unsigned)> unread =
      [&](unsigned tensor, int64_t axis, unsigned state) -> bool {
    if (llvm::is_contained(graph.outputs, tensor) ||
        llvm::is_contained(graph.inputs, tensor))
      return false;
    for (auto [reader, operand] : readers[tensor]) {
      const NodeRecord &node = graph.nodes[reader];
      if (isDeviceAlias(graph, node)) {
        std::optional<int64_t> moved = aliasAxis(shapeOf(graph, tensor), axis,
                                                 shapeOf(graph, node.output));
        if (!moved || !unread(node.output, *moved, state))
          return false;
        continue;
      }
      if (operand != 0 || extents.skippedRows[reader] != state ||
          axis != graph.tensors[tensor].type.getRank() - 1)
        return false;
    }
    return true;
  };

  for (auto [nodeIndex, node] : llvm::enumerate(graph.nodes)) {
    if (isDeviceAlias(graph, node))
      continue;
    if (const auto &zero = extents.zero[node.output]) {
      // The Concat of the past keeps its rows; writing them as the constant
      // is the same as copying them.
      extents.constantPrefix[nodeIndex] = zero;
      continue;
    }
    if (!isDeviceSoftmaxNode(graph, node, DeviceKernelLimits::Synthesized))
      continue;
    int64_t axis = normalizedAxis(integerAttribute(node, "axis", -1),
                                  graph.tensors[node.output].type.getRank());
    // The state of the first MatMul reader that skips rows; every reader
    // must then skip the rows of that state.
    std::optional<unsigned> state;
    std::function<void(unsigned)> find = [&](unsigned tensor) {
      for (auto [reader, operand] : readers[tensor]) {
        (void)operand;
        if (state)
          return;
        if (isDeviceAlias(graph, graph.nodes[reader]))
          find(graph.nodes[reader].output);
        else
          state = extents.skippedRows[reader];
      }
    };
    find(node.output);
    if (state && unread(node.output, axis, *state))
      extents.constantPrefix[nodeIndex] =
          DeviceConstantPrefix{{*state, axis}, 0u};
  }
  return extents;
}
} // namespace vrc
