// RUN: not %vrc-opt --split-input-file --udoncl-bound-onnx-shape-operands %s 2>&1 | %FileCheck %s

// The positions counted from the end of the extent (j - past) are not the
// leading block of their value at the bound.
module attributes {udoncl.dimension_bounds = {past = 3 : i64}} {
  func.func @main_graph(
      %past: tensor<?xf32> {udoncl.dim_exprs = [{axis = 0 : i64, offset = 0 : i64, symbol = "past"}]})
      -> tensor<?xf32> {
    %length_vector = "onnx.Dim"(%past) {axis = 0 : si64} : (tensor<?xf32>) -> tensor<1xi64>
    %axis = onnx.Constant dense<0> : tensor<1xi64>
    %length = "onnx.Squeeze"(%length_vector, %axis) : (tensor<1xi64>, tensor<1xi64>) -> tensor<i64>
    %zero = onnx.Constant dense<0> : tensor<i64>
    %step = onnx.Constant dense<1> : tensor<i64>
    %iota = "onnx.Range"(%zero, %length, %step) : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<?xi64>
    %down = "onnx.Sub"(%iota, %length) : (tensor<?xi64>, tensor<i64>) -> tensor<?xi64>
    %down_f = "onnx.Cast"(%down) {saturate = 1 : si64, to = f32} : (tensor<?xi64>) -> tensor<?xf32>
    %sum = "onnx.Add"(%past, %down_f) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    "onnx.Return"(%sum) : (tensor<?xf32>) -> ()
  }
}

// CHECK: error: ONNX operation onnx.Cast computes data that depends on 'past' beyond a bounded prefix

// -----

// An Expand that broadcasts run-time data to the extent of a bounded
// dimension has no extent-free shape operand.
module attributes {udoncl.dimension_bounds = {past = 3 : i64}} {
  func.func @main_graph(
      %value: tensor<1xf32>,
      %past: tensor<?xf32> {udoncl.dim_exprs = [{axis = 0 : i64, offset = 0 : i64, symbol = "past"}]})
      -> tensor<?xf32> {
    %length = "onnx.Dim"(%past) {axis = 0 : si64} : (tensor<?xf32>) -> tensor<1xi64>
    %expanded = "onnx.Expand"(%value, %length) : (tensor<1xf32>, tensor<1xi64>) -> tensor<?xf32>
    "onnx.Return"(%expanded) : (tensor<?xf32>) -> ()
  }
}

// CHECK: error: ONNX operation onnx.Expand has a shape operand that depends on a bounded dimension: it broadcasts to an extent of a bounded dimension
