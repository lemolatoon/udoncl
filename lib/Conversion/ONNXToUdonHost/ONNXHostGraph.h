#ifndef VRC_CONVERSION_ONNXTOUDONHOST_ONNXHOSTGRAPH_H
#define VRC_CONVERSION_ONNXTOUDONHOST_ONNXHOSTGRAPH_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include <string>

namespace vrc {
struct ONNXHostTensor {
  mlir::RankedTensorType type;
  mlir::DenseElementsAttr initialValue;
  std::string symbol;
  std::string name;
  int64_t staticResourceId = -1;
  int64_t staticByteLength = 0;
};

struct ONNXHostNode {
  std::string operation;
  std::string provider;
  llvm::SmallVector<int64_t> inputs;
  unsigned output;
  mlir::DictionaryAttr attributes;
};

struct ONNXHostSequenceState {
  unsigned inputIndex;
  unsigned outputIndex;
  int64_t axis;
  int64_t step;
  int64_t capacity;
};

struct ONNXHostGraph {
  llvm::SmallVector<ONNXHostTensor> tensors;
  llvm::SmallVector<unsigned> inputs;
  llvm::SmallVector<unsigned> outputs;
  llvm::SmallVector<ONNXHostNode> nodes;
  llvm::SmallVector<ONNXHostSequenceState> sequenceStates;
};

using TensorRecord = ONNXHostTensor;
using NodeRecord = ONNXHostNode;
using SequenceStateRecord = ONNXHostSequenceState;
using GraphRecord = ONNXHostGraph;

inline constexpr llvm::StringLiteral onnxHostGraphValidatedAttr =
    "udoncl.onnx_host_graph_validated";
inline constexpr llvm::StringLiteral hostGraphValidatedAttr =
    onnxHostGraphValidatedAttr;

mlir::FailureOr<ONNXHostGraph> readONNXHostGraph(mlir::ModuleOp module);
} // namespace vrc

#endif
