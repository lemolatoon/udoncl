// Fusion groups of the ONNX Device graph. A group is a tree of per-element
// and re-indexing nodes that one synthesized gather kernel evaluates: only
// the root's result is written to a buffer, the results of the interior
// nodes exist only inside that kernel.
#ifndef VRC_CONVERSION_ONNXTOUDONHOST_ONNXHOSTFUSION_H
#define VRC_CONVERSION_ONNXTOUDONHOST_ONNXHOSTFUSION_H

#include "ONNXHostGraph.h"

#include "mlir/Support/LLVM.h"

#include <optional>

namespace vrc {
// The cost model of a group. It is evaluated on the expression as a tree: a
// node or buffer with several readers inside the group counts once per
// reader. The synthesized kernel shares the Load of operands that read the
// same element, so its real cost can only be lower.

/// Distinct buffers one fused kernel reads. The gather profile binds at most
/// 13 input textures; a group stays well below that.
inline constexpr unsigned deviceFusionMaxLeafBuffers = 8;
/// Loads one fragment may execute. A Concat executes one of its sides, so
/// this bounds the longest path, not the Loads in the kernel text.
inline constexpr unsigned deviceFusionMaxPathLoads = 16;
/// Nodes of the expression tree of one kernel, which bounds the shader size
/// and the duplication of nodes with several readers.
inline constexpr unsigned deviceFusionMaxExpressionNodes = 64;
/// Loads of the operand expression of a reduction root. The root evaluates
/// it once per element of the reduced axis in each of its passes, as the
/// library kernel loads its input, so this bounds the kernel to a multiple
/// of the library kernel's Loads.
inline constexpr unsigned deviceFusionMaxReductionOperandLoads = 2;

enum class DeviceFusionKind {
  Add,
  Mul,
  Relu,
  Sigmoid,
  Tanh,
  Cos,
  Sin,
  Neg,
  Slice,
  Concat,
  Expand,
  Transpose,
  Alias,
  /// A reduction along one axis. It is only ever the root of a group: its
  /// kernel evaluates the operand expression at every element of the axis.
  Softmax,
};

/// True for kinds that only a group root may have.
inline bool isDeviceFusionReduction(DeviceFusionKind kind) {
  return kind == DeviceFusionKind::Softmax;
}

/// How a fused kernel can evaluate `node`, or nothing when it cannot: the
/// node is not one of the f32 per-element or re-indexing operations the
/// Device library implements, or its shapes are outside their contract.
std::optional<DeviceFusionKind> deviceFusionKind(const GraphRecord &graph,
                                                 const NodeRecord &node);

/// Leading inputs of a node of `kind` that are tensors of values; the
/// remaining inputs (Slice bounds, shapes) are static parameters.
unsigned deviceFusionValueOperands(DeviceFusionKind kind);

/// One value operand of a node of a fusion group.
struct DeviceFusionOperand {
  enum Kind {
    /// The result of another node of the group; `node` is that node.
    Interior,
    /// A known f32 scalar; `immediate` is its value.
    Immediate,
    /// An existing buffer; `tensor` identifies it across metadata aliases.
    Leaf,
  };
  Kind kind;
  unsigned node = 0;
  float immediate = 0.0f;
  unsigned tensor = 0;
};

struct DeviceFusionCost {
  unsigned leafBuffers = 0;
  /// Loads on the path of one fragment; for a reduction root, the Loads of
  /// one evaluation of its operand.
  unsigned pathLoads = 0;
  unsigned expressionNodes = 0;
  bool reduction = false;

  bool withinCaps() const {
    return leafBuffers <= deviceFusionMaxLeafBuffers &&
           pathLoads <= (reduction ? deviceFusionMaxReductionOperandLoads
                                   : deviceFusionMaxPathLoads) &&
           expressionNodes <= deviceFusionMaxExpressionNodes;
  }
};

/// The fusion groups of a graph: `roots` maps every node to the root of its
/// group, and a node outside every group to itself.
class DeviceFusion {
public:
  DeviceFusion(const GraphRecord &graph, llvm::ArrayRef<unsigned> roots);

  /// Forms the groups of `graph`. With `enabled` false, or for a graph the
  /// Device cannot run, every node is its own root.
  ///
  /// Groups are formed greedily from the last node to the first. Each node
  /// that is still its own root and is fusible, other than a metadata alias,
  /// grows a group breadth-first over the producers of its value operands. A
  /// producer joins when it is fusible, its result is not a graph input or
  /// output (which includes sequence state), it is not a reduction (which is
  /// only ever a root), every reader of its result is already in the group
  /// and reads it as a value, and the group stays
  /// within the caps above. A producer that would exceed a cap stays
  /// materialized for good. A group whose interior nodes are all metadata
  /// aliases saves no dispatch and is dissolved.
  static llvm::SmallVector<unsigned> plan(const GraphRecord &graph,
                                          bool enabled);

  /// True if `roots` describes groups a fused kernel can evaluate: every
  /// node of a group is fusible, the root is not a metadata alias and comes
  /// last, and only nodes of the group read the result of an interior node,
  /// as a value.
  static bool isValid(const GraphRecord &graph, llvm::ArrayRef<unsigned> roots);

  /// True if `nodeIndex` is the root of a group with interior nodes.
  bool isRoot(unsigned nodeIndex) const { return rootOfGroup[nodeIndex]; }
  /// True if the result of `nodeIndex` exists only inside its root's kernel.
  bool isInterior(unsigned nodeIndex) const {
    return roots[nodeIndex] != nodeIndex;
  }

  /// Value operand `operand` of `nodeIndex`, a node of a group. An f32 scalar
  /// constant is an Immediate only next to a non-constant operand of an Add
  /// or Mul, so that no operation of the kernel has only constant operands
  /// and nothing is evaluated at compile time instead of on the Device.
  DeviceFusionOperand operand(unsigned nodeIndex, unsigned operand) const;

  DeviceFusionCost cost(unsigned rootNode) const;

  /// The distinct Leaf tensors of the group rooted at `rootNode`, in the
  /// order a depth-first walk of the expression first reads them.
  llvm::SmallVector<unsigned> leaves(unsigned rootNode) const;

  const GraphRecord &graph;
  llvm::SmallVector<unsigned> roots;

private:
  DeviceFusionCost walk(unsigned rootNode,
                        llvm::SmallVectorImpl<unsigned> &leaves,
                        unsigned nodeLimit) const;

  /// Node that writes each tensor, or -1.
  llvm::SmallVector<int64_t> producers;
  llvm::SmallVector<bool> rootOfGroup;
};
} // namespace vrc

#endif
