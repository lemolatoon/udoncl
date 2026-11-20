// RUN: %vrc-opt --udoncl-fold-onnx-repeated-matmul-operand %s | %FileCheck %s

// Grouped-query attention: 2 key/value heads, each repeated for 3 query
// heads, multiply 6 query matrices of 2 rows. The MatMul reads the 2 heads
// directly; the 3 query matrices of a group become 6 rows of one matrix.

module {
  func.func @grouped(%probabilities: tensor<1x6x2x5xf32>,
                     %values: tensor<1x2x5x4xf32>) -> tensor<1x6x2x4xf32> {
    %axis = onnx.Constant dense<2> : tensor<1xi64>
    %expanded_shape = onnx.Constant dense<[1, 2, 3, 5, 4]> : tensor<5xi64>
    %heads_shape = onnx.Constant dense<[1, 6, 5, 4]> : tensor<4xi64>
    %unsqueezed = "onnx.Unsqueeze"(%values, %axis)
        : (tensor<1x2x5x4xf32>, tensor<1xi64>) -> tensor<1x2x1x5x4xf32>
    %expanded = "onnx.Expand"(%unsqueezed, %expanded_shape)
        : (tensor<1x2x1x5x4xf32>, tensor<5xi64>) -> tensor<1x2x3x5x4xf32>
    %heads = "onnx.Reshape"(%expanded, %heads_shape) {allowzero = 0 : si64}
        : (tensor<1x2x3x5x4xf32>, tensor<4xi64>) -> tensor<1x6x5x4xf32>
    %context = "onnx.MatMul"(%probabilities, %heads) {onnx_node_name = "context"}
        : (tensor<1x6x2x5xf32>, tensor<1x6x5x4xf32>) -> tensor<1x6x2x4xf32>
    "onnx.Return"(%context) : (tensor<1x6x2x4xf32>) -> ()
  }

// CHECK-LABEL: func.func @grouped
// CHECK-SAME: (%[[P:.*]]: tensor<1x6x2x5xf32>, %[[V:.*]]: tensor<1x2x5x4xf32>)
// CHECK-DAG: %[[ROWS_SHAPE:.*]] = arith.constant dense<[1, 2, 6, 5]> : tensor<4xi64>
// CHECK-DAG: %[[MATRIX_SHAPE:.*]] = arith.constant dense<[1, 2, 5, 4]> : tensor<4xi64>
// CHECK-DAG: %[[RESULT_SHAPE:.*]] = arith.constant dense<[1, 6, 2, 4]> : tensor<4xi64>
// CHECK: %[[U:.*]] = "onnx.Unsqueeze"(%[[V]]
// CHECK: %[[ROWS:.*]] = "onnx.Reshape"(%[[P]], %[[ROWS_SHAPE]]){{.*}} : (tensor<1x6x2x5xf32>, tensor<4xi64>) -> tensor<1x2x6x5xf32>
// CHECK: %[[MATRICES:.*]] = "onnx.Reshape"(%[[U]], %[[MATRIX_SHAPE]]){{.*}} : (tensor<1x2x1x5x4xf32>, tensor<4xi64>) -> tensor<1x2x5x4xf32>
// CHECK: %[[PRODUCT:.*]] = "onnx.MatMul"(%[[ROWS]], %[[MATRICES]]) {onnx_node_name = "context"} : (tensor<1x2x6x5xf32>, tensor<1x2x5x4xf32>) -> tensor<1x2x6x4xf32>
// CHECK: %[[RESULT:.*]] = "onnx.Reshape"(%[[PRODUCT]], %[[RESULT_SHAPE]]){{.*}} : (tensor<1x2x6x4xf32>, tensor<4xi64>) -> tensor<1x6x2x4xf32>
// CHECK-NOT: onnx.Expand
// CHECK: onnx.Return %[[RESULT]]

  // A MatMul whose right operand repeats along an axis other than the one
  // merged into the batch, or whose left operand broadcasts, is kept.
  func.func @kept(%probabilities: tensor<1x1x2x5xf32>,
                  %values: tensor<1x2x1x5x4xf32>,
                  %columns: tensor<1x2x3x5x1xf32>)
      -> (tensor<1x6x2x4xf32>, tensor<1x6x2x4xf32>) {
    %expanded_shape = onnx.Constant dense<[1, 2, 3, 5, 4]> : tensor<5xi64>
    %heads_shape = onnx.Constant dense<[1, 6, 5, 4]> : tensor<4xi64>
    %expanded = "onnx.Expand"(%values, %expanded_shape)
        : (tensor<1x2x1x5x4xf32>, tensor<5xi64>) -> tensor<1x2x3x5x4xf32>
    %heads = "onnx.Reshape"(%expanded, %heads_shape) {allowzero = 0 : si64}
        : (tensor<1x2x3x5x4xf32>, tensor<4xi64>) -> tensor<1x6x5x4xf32>
    %broadcast_left = "onnx.MatMul"(%probabilities, %heads)
        : (tensor<1x1x2x5xf32>, tensor<1x6x5x4xf32>) -> tensor<1x6x2x4xf32>
    %wide = "onnx.Expand"(%columns, %expanded_shape)
        : (tensor<1x2x3x5x1xf32>, tensor<5xi64>) -> tensor<1x2x3x5x4xf32>
    %wide_heads = "onnx.Reshape"(%wide, %heads_shape) {allowzero = 0 : si64}
        : (tensor<1x2x3x5x4xf32>, tensor<4xi64>) -> tensor<1x6x5x4xf32>
    %full = "onnx.Concat"(%probabilities, %probabilities, %probabilities,
                          %probabilities, %probabilities, %probabilities)
        {axis = 1 : si64}
        : (tensor<1x1x2x5xf32>, tensor<1x1x2x5xf32>, tensor<1x1x2x5xf32>,
           tensor<1x1x2x5xf32>, tensor<1x1x2x5xf32>, tensor<1x1x2x5xf32>)
        -> tensor<1x6x2x5xf32>
    %columns_product = "onnx.MatMul"(%full, %wide_heads)
        : (tensor<1x6x2x5xf32>, tensor<1x6x5x4xf32>) -> tensor<1x6x2x4xf32>
    "onnx.Return"(%broadcast_left, %columns_product)
        : (tensor<1x6x2x4xf32>, tensor<1x6x2x4xf32>) -> ()
  }

// CHECK-LABEL: func.func @kept
// CHECK: "onnx.Expand"
// CHECK: "onnx.MatMul"({{.*}}) : (tensor<1x1x2x5xf32>, tensor<1x6x5x4xf32>)
// CHECK: "onnx.Expand"
// CHECK: "onnx.MatMul"({{.*}}) : (tensor<1x6x2x5xf32>, tensor<1x6x5x4xf32>)

  // The key operand of attention transposes and scales the repeated heads
  // before the MatMul. The transpose and the scale move in front of the
  // repetition, element for element, so the repetition folds as well: the
  // MatMul reads 2 scaled, transposed key heads instead of 6.
  func.func @keys(%queries: tensor<1x6x1x4xf32>,
                  %keys: tensor<1x2x5x4xf32>) -> tensor<1x6x1x5xf32> {
    %axis = onnx.Constant dense<2> : tensor<1xi64>
    %expanded_shape = onnx.Constant dense<[1, 2, 3, 5, 4]> : tensor<5xi64>
    %heads_shape = onnx.Constant dense<[1, 6, 5, 4]> : tensor<4xi64>
    %scale = onnx.Constant dense<0.5> : tensor<1xf32>
    %unsqueezed = "onnx.Unsqueeze"(%keys, %axis)
        : (tensor<1x2x5x4xf32>, tensor<1xi64>) -> tensor<1x2x1x5x4xf32>
    %expanded = "onnx.Expand"(%unsqueezed, %expanded_shape)
        : (tensor<1x2x1x5x4xf32>, tensor<5xi64>) -> tensor<1x2x3x5x4xf32>
    %heads = "onnx.Reshape"(%expanded, %heads_shape) {allowzero = 0 : si64}
        : (tensor<1x2x3x5x4xf32>, tensor<4xi64>) -> tensor<1x6x5x4xf32>
    %transposed = "onnx.Transpose"(%heads) {perm = [0, 1, 3, 2]}
        : (tensor<1x6x5x4xf32>) -> tensor<1x6x4x5xf32>
    %scaled = "onnx.Mul"(%transposed, %scale)
        : (tensor<1x6x4x5xf32>, tensor<1xf32>) -> tensor<1x6x4x5xf32>
    %scores = "onnx.MatMul"(%queries, %scaled) {onnx_node_name = "scores"}
        : (tensor<1x6x1x4xf32>, tensor<1x6x4x5xf32>) -> tensor<1x6x1x5xf32>
    "onnx.Return"(%scores) : (tensor<1x6x1x5xf32>) -> ()
  }

// CHECK-LABEL: func.func @keys
// CHECK-SAME: (%[[Q:.*]]: tensor<1x6x1x4xf32>, %[[K:.*]]: tensor<1x2x5x4xf32>)
// CHECK: %[[SCALE:.*]] = onnx.Constant dense<5.000000e-01> : tensor<1xf32>
// CHECK: %[[U:.*]] = "onnx.Unsqueeze"(%[[K]]
// CHECK: %[[TURNED:.*]] = "onnx.Transpose"(%[[U]]) <{perm = [0, 1, 2, 4, 3]}> : (tensor<1x2x1x5x4xf32>) -> tensor<1x2x1x4x5xf32>
// CHECK: %[[SCALED:.*]] = "onnx.Mul"(%[[TURNED]], %[[SCALE]]) : (tensor<1x2x1x4x5xf32>, tensor<1xf32>) -> tensor<1x2x1x4x5xf32>
// CHECK: %[[ROWS:.*]] = "onnx.Reshape"(%[[Q]], %{{.*}}){{.*}} -> tensor<1x2x3x4xf32>
// CHECK: %[[MATRICES:.*]] = "onnx.Reshape"(%[[SCALED]], %{{.*}}){{.*}} -> tensor<1x2x4x5xf32>
// CHECK: %[[PRODUCT:.*]] = "onnx.MatMul"(%[[ROWS]], %[[MATRICES]]) {onnx_node_name = "scores"} : (tensor<1x2x3x4xf32>, tensor<1x2x4x5xf32>) -> tensor<1x2x3x5xf32>
// CHECK: %[[RESULT:.*]] = "onnx.Reshape"(%[[PRODUCT]], %{{.*}}){{.*}} -> tensor<1x6x1x5xf32>
// CHECK-NOT: onnx.Expand
// CHECK: onnx.Return %[[RESULT]]
}
