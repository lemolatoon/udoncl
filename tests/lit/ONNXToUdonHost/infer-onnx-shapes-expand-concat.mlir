// RUN: %vrc-opt --udoncl-infer-onnx-shapes %s | %FileCheck %s

// Statically known integer Expand results and Concat results along a
// non-leading axis are folded to constants, as other integer shape values are.

module {
  func.func @main_graph(%dynamic: tensor<2x1xi64>)
      -> (tensor<2x3x2xi64>, tensor<2x3x2xi64>) {
    %rows = "onnx.Constant"() {value = dense<[[0], [1]]> : tensor<2x1xi64>}
        : () -> tensor<2x1xi64>
    %columns = "onnx.Constant"() {value = dense<[4, 5, 6]> : tensor<3xi64>}
        : () -> tensor<3xi64>
    %shape = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64>}
        : () -> tensor<2xi64>
    %row_grid = "onnx.Expand"(%rows, %shape)
        : (tensor<2x1xi64>, tensor<2xi64>) -> tensor<2x3xi64>
    %column_grid = "onnx.Expand"(%columns, %shape)
        : (tensor<3xi64>, tensor<2xi64>) -> tensor<2x3xi64>
    %axis = "onnx.Constant"() {value = dense<2> : tensor<1xi64>}
        : () -> tensor<1xi64>
    %row_index = "onnx.Unsqueeze"(%row_grid, %axis)
        : (tensor<2x3xi64>, tensor<1xi64>) -> tensor<2x3x1xi64>
    %column_index = "onnx.Unsqueeze"(%column_grid, %axis)
        : (tensor<2x3xi64>, tensor<1xi64>) -> tensor<2x3x1xi64>
    %indices = "onnx.Concat"(%row_index, %column_index) {axis = -1 : si64}
        : (tensor<2x3x1xi64>, tensor<2x3x1xi64>) -> tensor<2x3x2xi64>
    %dynamic_grid = "onnx.Expand"(%dynamic, %shape)
        : (tensor<2x1xi64>, tensor<2xi64>) -> tensor<2x3xi64>
    %dynamic_index = "onnx.Unsqueeze"(%dynamic_grid, %axis)
        : (tensor<2x3xi64>, tensor<1xi64>) -> tensor<2x3x1xi64>
    %mixed = "onnx.Concat"(%row_index, %dynamic_index) {axis = 2 : si64}
        : (tensor<2x3x1xi64>, tensor<2x3x1xi64>) -> tensor<2x3x2xi64>
    "onnx.Return"(%indices, %mixed)
        : (tensor<2x3x2xi64>, tensor<2x3x2xi64>) -> ()
  }
}

// CHECK-LABEL: func.func @main_graph
// CHECK-DAG: %[[INDICES:.*]] = arith.constant dense<{{\[\[\[}}0, 4], [0, 5], [0, 6]], {{\[\[}}1, 4], [1, 5], [1, 6]]]> : tensor<2x3x2xi64>
// CHECK-DAG: %[[DYNAMIC:.*]] = "onnx.Expand"(%arg0
// CHECK-DAG: %[[MIXED:.*]] = "onnx.Concat"
// CHECK: onnx.Return %[[INDICES]], %[[MIXED]]
