#include "ONNXHostFusion.h"

#include "ONNXHostPlan.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>
#include <numeric>

using namespace mlir;

namespace vrc {
namespace {
/// Words of the largest tensor a fused kernel writes or reads as a Device
/// buffer.
constexpr int64_t fusionSurfaceWords = deviceSynthesizedSurfaceWords;

bool isStaticF32(const GraphRecord &graph, int64_t tensorIndex) {
  if (tensorIndex < 0 ||
      static_cast<size_t>(tensorIndex) >= graph.tensors.size())
    return false;
  RankedTensorType type = graph.tensors[tensorIndex].type;
  return type.getElementType().isF32() && type.hasStaticShape() &&
         type.getNumElements() >= 1 &&
         type.getNumElements() <= fusionSurfaceWords;
}

/// NumPy broadcasting of `input` to `output` on the real shapes. The library
/// kernels broadcast on shapes folded to four dimensions; wherever both hold
/// they select the same element.
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
} // namespace

unsigned deviceFusionValueOperands(DeviceFusionKind kind) {
  switch (kind) {
  case DeviceFusionKind::Add:
  case DeviceFusionKind::Mul:
  case DeviceFusionKind::Concat:
    return 2;
  default:
    return 1;
  }
}

std::optional<DeviceFusionKind> deviceFusionKind(const GraphRecord &graph,
                                                 const NodeRecord &node) {
  static constexpr std::pair<StringLiteral, DeviceFusionKind> kinds[] = {
      {"Add", DeviceFusionKind::Add},
      {"Mul", DeviceFusionKind::Mul},
      {"Relu", DeviceFusionKind::Relu},
      {"Sigmoid", DeviceFusionKind::Sigmoid},
      {"Tanh", DeviceFusionKind::Tanh},
      {"Cos", DeviceFusionKind::Cos},
      {"Sin", DeviceFusionKind::Sin},
      {"Neg", DeviceFusionKind::Neg},
      {"Slice", DeviceFusionKind::Slice},
      {"Concat", DeviceFusionKind::Concat},
      {"Expand", DeviceFusionKind::Expand},
      {"Transpose", DeviceFusionKind::Transpose},
      {"Softmax", DeviceFusionKind::Softmax},
  };
  std::optional<DeviceFusionKind> kind;
  if (isMetadataOnlyAlias(node))
    kind = DeviceFusionKind::Alias;
  for (const auto &[operation, candidate] : kinds)
    if (node.operation == operation)
      kind = candidate;
  // supportsDeviceNode checks the provider and the limits of the library
  // kernel, which a fused kernel replaces element for element, or of a
  // synthesized kernel beyond them.
  if (!kind || !supportsDeviceNode(graph, node) ||
      !isStaticF32(graph, node.output))
    return std::nullopt;
  unsigned operands = deviceFusionValueOperands(*kind);
  if (node.inputs.size() < operands)
    return std::nullopt;
  RankedTensorType output = graph.tensors[node.output].type;
  for (unsigned operand = 0; operand != operands; ++operand) {
    if (!isStaticF32(graph, node.inputs[operand]))
      return std::nullopt;
    RankedTensorType input = graph.tensors[node.inputs[operand]].type;
    switch (*kind) {
    case DeviceFusionKind::Add:
    case DeviceFusionKind::Mul:
    case DeviceFusionKind::Expand:
      if (!broadcastsTo(input, output))
        return std::nullopt;
      break;
    case DeviceFusionKind::Slice:
    case DeviceFusionKind::Concat:
    case DeviceFusionKind::Transpose:
      // Their Device shape helpers have checked the shapes.
      break;
    default:
      if (input.getNumElements() != output.getNumElements())
        return std::nullopt;
      break;
    }
  }
  return kind;
}

DeviceFusion::DeviceFusion(const GraphRecord &graph, ArrayRef<unsigned> roots)
    : graph(graph), roots(roots), producers(graph.tensors.size(), -1),
      rootOfGroup(graph.nodes.size(), false) {
  for (auto [nodeIndex, node] : llvm::enumerate(graph.nodes))
    producers[node.output] = static_cast<int64_t>(nodeIndex);
  for (unsigned nodeIndex = 0; nodeIndex != graph.nodes.size(); ++nodeIndex)
    if (roots[nodeIndex] != nodeIndex)
      rootOfGroup[roots[nodeIndex]] = true;
}

bool DeviceFusion::isValid(const GraphRecord &graph, ArrayRef<unsigned> roots) {
  if (roots.size() != graph.nodes.size())
    return false;
  SmallVector<int64_t> interiorRoot(graph.tensors.size(), -1);
  for (auto [nodeIndex, node] : llvm::enumerate(graph.nodes)) {
    unsigned root = roots[nodeIndex];
    if (root == nodeIndex)
      continue;
    std::optional<DeviceFusionKind> kind = deviceFusionKind(graph, node);
    if (root < nodeIndex || root >= roots.size() || roots[root] != root ||
        !kind || isDeviceFusionReduction(*kind) ||
        llvm::is_contained(graph.inputs, node.output) ||
        llvm::is_contained(graph.outputs, node.output))
      return false;
    std::optional<DeviceFusionKind> rootKind =
        deviceFusionKind(graph, graph.nodes[root]);
    if (!rootKind || *rootKind == DeviceFusionKind::Alias)
      return false;
    interiorRoot[node.output] = root;
  }
  for (auto [nodeIndex, node] : llvm::enumerate(graph.nodes)) {
    std::optional<DeviceFusionKind> kind = deviceFusionKind(graph, node);
    for (auto [index, input] : llvm::enumerate(node.inputs))
      if (input >= 0 && interiorRoot[input] >= 0 &&
          (interiorRoot[input] != roots[nodeIndex] || !kind ||
           index >= deviceFusionValueOperands(*kind)))
        return false;
  }
  return true;
}

DeviceFusionOperand DeviceFusion::operand(unsigned nodeIndex,
                                          unsigned operand) const {
  const NodeRecord &node = graph.nodes[nodeIndex];
  unsigned tensor = static_cast<unsigned>(node.inputs[operand]);
  int64_t producer = producers[tensor];
  if (producer >= 0 && roots[producer] == roots[nodeIndex] &&
      static_cast<unsigned>(producer) != roots[nodeIndex])
    return {DeviceFusionOperand::Interior, static_cast<unsigned>(producer)};
  if ((node.operation == "Add" || node.operation == "Mul") &&
      node.inputs.size() == 2) {
    std::optional<float> value = scalarF32(graph.tensors[tensor]);
    if (value && !scalarF32(graph.tensors[node.inputs[1 - operand]]))
      return {DeviceFusionOperand::Immediate, 0, *value};
  }
  // A metadata alias outside the group shares the buffer of its input.
  while (producers[tensor] >= 0 &&
         isDeviceAlias(graph, graph.nodes[producers[tensor]]))
    tensor = static_cast<unsigned>(graph.nodes[producers[tensor]].inputs[0]);
  return {DeviceFusionOperand::Leaf, 0, 0.0f, tensor};
}

DeviceFusionCost DeviceFusion::walk(unsigned rootNode,
                                    SmallVectorImpl<unsigned> &leaves,
                                    unsigned nodeLimit) const {
  DeviceFusionCost cost;
  auto visit = [&](auto &&self, unsigned nodeIndex) -> unsigned {
    if (++cost.expressionNodes > nodeLimit)
      return 0;
    const NodeRecord &node = graph.nodes[nodeIndex];
    DeviceFusionKind kind = *deviceFusionKind(graph, node);
    unsigned loads[2] = {0, 0};
    for (unsigned index = 0; index != deviceFusionValueOperands(kind);
         ++index) {
      DeviceFusionOperand value = operand(nodeIndex, index);
      if (value.kind == DeviceFusionOperand::Interior) {
        loads[index] = self(self, value.node);
      } else if (value.kind == DeviceFusionOperand::Leaf) {
        loads[index] = 1;
        if (!llvm::is_contained(leaves, value.tensor))
          leaves.push_back(value.tensor);
      }
    }
    // A fragment reads one side of a Concat.
    return kind == DeviceFusionKind::Concat ? std::max(loads[0], loads[1])
                                            : loads[0] + loads[1];
  };
  // A reduction root evaluates its operand per element of the reduced axis;
  // its cap applies to one evaluation.
  cost.reduction =
      isDeviceFusionReduction(*deviceFusionKind(graph, graph.nodes[rootNode]));
  cost.pathLoads = visit(visit, rootNode);
  cost.leafBuffers = leaves.size();
  return cost;
}

DeviceFusionCost DeviceFusion::cost(unsigned rootNode) const {
  SmallVector<unsigned> leaves;
  // Past the cap the exact count no longer matters, and a node with several
  // readers could otherwise multiply the walk.
  return walk(rootNode, leaves, deviceFusionMaxExpressionNodes);
}

SmallVector<unsigned> DeviceFusion::leaves(unsigned rootNode) const {
  SmallVector<unsigned> leaves;
  walk(rootNode, leaves, std::numeric_limits<unsigned>::max());
  return leaves;
}

SmallVector<unsigned> DeviceFusion::plan(const GraphRecord &graph,
                                         bool enabled) {
  SmallVector<unsigned> identity(graph.nodes.size());
  std::iota(identity.begin(), identity.end(), 0u);
  if (!enabled || !supportsDeviceGraph(graph))
    return identity;
  DeviceFusion fusion(graph, identity);
  SmallVector<unsigned> &roots = fusion.roots;

  SmallVector<std::optional<DeviceFusionKind>> kinds;
  kinds.reserve(graph.nodes.size());
  for (const NodeRecord &node : graph.nodes)
    kinds.push_back(deviceFusionKind(graph, node));
  SmallVector<SmallVector<unsigned, 2>> readers(graph.tensors.size());
  for (auto [nodeIndex, node] : llvm::enumerate(graph.nodes))
    for (int64_t input : node.inputs)
      if (input >= 0 && !llvm::is_contained(readers[input], nodeIndex))
        readers[input].push_back(nodeIndex);
  SmallVector<bool> external(graph.tensors.size(), false);
  for (unsigned tensor : graph.inputs)
    external[tensor] = true;
  for (unsigned tensor : graph.outputs)
    external[tensor] = true;

  SmallVector<bool> rejected(graph.nodes.size(), false);
  for (unsigned root = graph.nodes.size(); root-- > 0;) {
    if (roots[root] != root || !kinds[root] ||
        *kinds[root] == DeviceFusionKind::Alias)
      continue;
    auto joins = [&](unsigned producer) {
      const NodeRecord &node = graph.nodes[producer];
      if (roots[producer] != producer || rejected[producer] ||
          !kinds[producer] || isDeviceFusionReduction(*kinds[producer]) ||
          external[node.output])
        return false;
      for (unsigned reader : readers[node.output]) {
        if (roots[reader] != root)
          return false;
        ArrayRef<int64_t> inputs = graph.nodes[reader].inputs;
        if (llvm::is_contained(
                inputs.drop_front(deviceFusionValueOperands(*kinds[reader])),
                static_cast<int64_t>(node.output)))
          return false;
      }
      roots[producer] = root;
      if (fusion.cost(root).withinCaps())
        return true;
      roots[producer] = producer;
      rejected[producer] = true;
      return false;
    };
    SmallVector<unsigned> members{root};
    for (bool grown = true; grown;) {
      grown = false;
      // `members` grows while it is walked: breadth-first from the root.
      for (size_t index = 0; index != members.size(); ++index) {
        unsigned member = members[index];
        for (unsigned operand = 0;
             operand != deviceFusionValueOperands(*kinds[member]); ++operand) {
          int64_t producer =
              fusion.producers[graph.nodes[member].inputs[operand]];
          if (producer >= 0 && joins(static_cast<unsigned>(producer))) {
            members.push_back(static_cast<unsigned>(producer));
            grown = true;
          }
        }
      }
    }
    if (llvm::none_of(ArrayRef<unsigned>(members).drop_front(),
                      [&](unsigned member) {
                        return *kinds[member] != DeviceFusionKind::Alias;
                      }))
      for (unsigned member : members)
        roots[member] = member;
  }
  return roots;
}
} // namespace vrc
