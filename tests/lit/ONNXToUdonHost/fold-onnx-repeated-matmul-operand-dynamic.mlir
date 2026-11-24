// RUN: %vrc-opt --udoncl-fold-onnx-repeated-matmul-operand %s | %FileCheck %s

// Grouped-query attention over a past of a bounded length: the repeated key
// (dynamic N after the transpose) and value (dynamic K) fold into the rows of
// the left operand as with static extents; the shape operands use -1 (a
// Reshape) and 1 (an Expand) at the dynamic axis.

module {
  func.func @main_graph(%query: tensor<1x4x1x8xf32>, %key: tensor<1x2x?x8xf32>,
                        %probabilities: tensor<1x4x1x?xf32>, %value: tensor<1x2x?x8xf32>)
      -> (tensor<1x4x1x?xf32>, tensor<1x4x1x8xf32>) {
    %axis = onnx.Constant dense<2> : tensor<1xi64>
    %repeat = onnx.Constant dense<[1, 2, 2, 1, 8]> : tensor<5xi64>
    %heads = onnx.Constant dense<[1, 4, -1, 8]> : tensor<4xi64>
    %scale = onnx.Constant dense<0.5> : tensor<1xf32>
    %key_unsqueezed = "onnx.Unsqueeze"(%key, %axis)
        : (tensor<1x2x?x8xf32>, tensor<1xi64>) -> tensor<1x2x1x?x8xf32>
    %key_repeated = "onnx.Expand"(%key_unsqueezed, %repeat)
        : (tensor<1x2x1x?x8xf32>, tensor<5xi64>) -> tensor<1x2x2x?x8xf32>
    %key_heads = "onnx.Reshape"(%key_repeated, %heads)
        : (tensor<1x2x2x?x8xf32>, tensor<4xi64>) -> tensor<1x4x?x8xf32>
    %key_transposed = "onnx.Transpose"(%key_heads) {perm = [0, 1, 3, 2]}
        : (tensor<1x4x?x8xf32>) -> tensor<1x4x8x?xf32>
    %key_scaled = "onnx.Mul"(%key_transposed, %scale)
        : (tensor<1x4x8x?xf32>, tensor<1xf32>) -> tensor<1x4x8x?xf32>
    %scores = "onnx.MatMul"(%query, %key_scaled)
        : (tensor<1x4x1x8xf32>, tensor<1x4x8x?xf32>) -> tensor<1x4x1x?xf32>
    %value_unsqueezed = "onnx.Unsqueeze"(%value, %axis)
        : (tensor<1x2x?x8xf32>, tensor<1xi64>) -> tensor<1x2x1x?x8xf32>
    %value_repeated = "onnx.Expand"(%value_unsqueezed, %repeat)
        : (tensor<1x2x1x?x8xf32>, tensor<5xi64>) -> tensor<1x2x2x?x8xf32>
    %value_heads = "onnx.Reshape"(%value_repeated, %heads)
        : (tensor<1x2x2x?x8xf32>, tensor<4xi64>) -> tensor<1x4x?x8xf32>
    %attention = "onnx.MatMul"(%probabilities, %value_heads)
        : (tensor<1x4x1x?xf32>, tensor<1x4x?x8xf32>) -> tensor<1x4x1x8xf32>
    "onnx.Return"(%scores, %attention) : (tensor<1x4x1x?xf32>, tensor<1x4x1x8xf32>) -> ()
  }
}

// CHECK-LABEL: func.func @main_graph
// CHECK-DAG: %[[ROWS:.*]] = arith.constant dense<[1, 2, 2, 8]> : tensor<4xi64>
// CHECK-DAG: %[[KEYS:.*]] = arith.constant dense<[1, 2, 8, -1]> : tensor<4xi64>
// CHECK-DAG: %[[SCORES:.*]] = arith.constant dense<[1, 4, 1, -1]> : tensor<4xi64>
// CHECK-DAG: %[[PROBABILITY_ROWS:.*]] = arith.constant dense<[1, 2, 2, -1]> : tensor<4xi64>
// CHECK-DAG: %[[VALUES:.*]] = arith.constant dense<[1, 2, -1, 8]> : tensor<4xi64>
// CHECK: "onnx.Transpose"({{.*}}) {{.*}} : (tensor<1x2x1x?x8xf32>) -> tensor<1x2x1x8x?xf32>
// CHECK: "onnx.MatMul"({{.*}}) : (tensor<1x2x2x8xf32>, tensor<1x2x8x?xf32>) -> tensor<1x2x2x?xf32>
// CHECK: "onnx.MatMul"({{.*}}) : (tensor<1x2x2x?xf32>, tensor<1x2x?x8xf32>) -> tensor<1x2x2x8xf32>
// CHECK-NOT: onnx.Expand
