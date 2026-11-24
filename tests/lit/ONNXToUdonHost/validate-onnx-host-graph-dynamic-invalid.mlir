// RUN: not %vrc-opt --split-input-file --udoncl-validate-onnx-host-graph %s 2>&1 | %FileCheck %s

// An operator that reads a tensor with an axis of a bounded dimension must
// support its run-time extent.
module attributes {udoncl.dimension_bounds = {past = 7 : i64}} {
  func.func private @__udon_onnx_add(tensor<1x?xf32>, tensor<1x?xf32>)
      -> tensor<1x?xf32>
      attributes {udon.onnx.operation = "Add", udon.onnx.provider = "udonatenAdd"}

  func.func @main_graph(
      %past: tensor<1x?xf32> {onnx.name = "past", udoncl.dim_exprs = [{axis = 1 : i64, offset = 0 : i64, symbol = "past"}]})
      -> (tensor<1x?xf32> {onnx.name = "sum"}) {
    %sum = func.call @__udon_onnx_add(%past, %past) {udon.onnx.result_dim_expr = {axis = 1 : i64, offset = 0 : i64, symbol = "past"}}
        : (tensor<1x?xf32>, tensor<1x?xf32>) -> tensor<1x?xf32>
    return %sum : tensor<1x?xf32>
  }
}

// CHECK: error: ONNX operator Add does not support a dynamic extent on axis 1 (dimension past)

// -----

// A view that moves elements across the dynamic axis reads different words
// at different extents.
module attributes {udoncl.dimension_bounds = {past = 7 : i64}} {
  func.func private @__udon_onnx_reshape(tensor<2x?xf32>, tensor<2xi64>)
      -> tensor<?x2xf32>
      attributes {udon.onnx.operation = "Reshape", udon.onnx.provider = "udononnxReshape"}

  func.func @main_graph(
      %past: tensor<2x?xf32> {onnx.name = "past", udoncl.dim_exprs = [{axis = 1 : i64, offset = 0 : i64, symbol = "past"}]})
      -> (tensor<?x2xf32> {onnx.name = "turned"}) {
    %shape = arith.constant dense<[-1, 2]> : tensor<2xi64>
    %turned = func.call @__udon_onnx_reshape(%past, %shape) {udon.onnx.result_dim_expr = {axis = 0 : i64, offset = 0 : i64, symbol = "past"}}
        : (tensor<2x?xf32>, tensor<2xi64>) -> tensor<?x2xf32>
    return %turned : tensor<?x2xf32>
  }
}

// CHECK: error: ONNX operator Reshape does not support a dynamic extent on axis 0 (dimension past)

// -----

// A dynamic tensor must name its bounded dimension.
module attributes {udoncl.dimension_bounds = {past = 7 : i64}} {
  func.func @main_graph(%past: tensor<1x?xf32> {onnx.name = "past"})
      -> (tensor<1x?xf32> {onnx.name = "same"}) {
    return %past : tensor<1x?xf32>
  }
}

// CHECK: error: ONNX Host tensor requires a nonnegative static ranked shape
