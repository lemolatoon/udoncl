#ifndef VRC_CONVERSION_ONNXTOUDONHOST_ONNXHOSTGRAPH_H
#define VRC_CONVERSION_ONNXTOUDONHOST_ONNXHOSTGRAPH_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include <optional>
#include <string>

namespace vrc {
/// The one axis of a tensor whose extent depends on a bounded dimension: at
/// run time it is `symbol + offset`, at most `bound + offset`.
struct ONNXHostDynamicAxis {
  int64_t axis;
  std::string symbol;
  int64_t offset;
};

struct ONNXHostTensor {
  /// The tensor type; a dynamic axis has its largest extent (the bound of
  /// its dimension plus the offset), so every size planned from it holds the
  /// tensor at every extent.
  mlir::RankedTensorType type;
  mlir::DenseElementsAttr initialValue;
  std::string symbol;
  std::string name;
  int64_t staticResourceId = -1;
  int64_t staticByteLength = 0;
  /// Set for a tensor with an axis of a bounded dimension.
  std::optional<ONNXHostDynamicAxis> dynamic;
};

struct ONNXHostNode {
  std::string operation;
  std::string provider;
  llvm::SmallVector<int64_t> inputs;
  unsigned output;
  mlir::DictionaryAttr attributes;
  /// Where the provider call is, for diagnostics.
  std::optional<mlir::Location> location;
};

struct ONNXHostSequenceState {
  unsigned inputIndex;
  unsigned outputIndex;
  int64_t axis;
  int64_t step;
  int64_t capacity;
};

struct ONNXHostGraph {
  /// Every bounded dimension and its inclusive bound, in name order.
  llvm::SmallVector<std::pair<std::string, int64_t>> dimensionBounds;
  llvm::SmallVector<ONNXHostTensor> tensors;
  llvm::SmallVector<unsigned> inputs;
  llvm::SmallVector<unsigned> outputs;
  llvm::SmallVector<ONNXHostNode> nodes;
  llvm::SmallVector<ONNXHostSequenceState> sequenceStates;
};

using DynamicAxisRecord = ONNXHostDynamicAxis;
using TensorRecord = ONNXHostTensor;
using NodeRecord = ONNXHostNode;
using SequenceStateRecord = ONNXHostSequenceState;
using GraphRecord = ONNXHostGraph;

inline constexpr llvm::StringLiteral onnxHostGraphValidatedAttr =
    "udoncl.onnx_host_graph_validated";
inline constexpr llvm::StringLiteral hostGraphValidatedAttr =
    onnxHostGraphValidatedAttr;

/// Call attribute of an ONNX Host provider call whose result has an axis of a
/// bounded dimension: {axis, symbol, offset}.
inline constexpr llvm::StringLiteral onnxHostResultDimensionAttr =
    "udon.onnx.result_dim_expr";

mlir::FailureOr<ONNXHostGraph> readONNXHostGraph(mlir::ModuleOp module);

/// The inclusive bound of `symbol` in `graph`.
std::optional<int64_t> dimensionBound(const ONNXHostGraph &graph,
                                      llvm::StringRef symbol);
} // namespace vrc

#endif
