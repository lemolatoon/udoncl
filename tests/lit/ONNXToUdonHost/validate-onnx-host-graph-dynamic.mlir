// RUN: %vrc-opt --udoncl-validate-onnx-host-graph %s | %FileCheck %s

// A Host graph with a bounded dimension: the past input carries its extent
// expression, the provider result records its own, and a view that keeps the
// elements on either side of the dynamic axis supports every extent. A
// bounded prefix of a constant at the bound is a view as well.

module attributes {udoncl.dimension_bounds = {past = 7 : i64}} {
  func.func private @__udon_onnx_unsqueeze(tensor<1x2x?x4xf32>, tensor<1xi64>)
      -> tensor<1x2x1x?x4xf32>
      attributes {udon.onnx.operation = "Unsqueeze",
                  udon.onnx.provider = "udononnxUnsqueeze"}
  func.func private @__udon_onnx_boundedprefix(tensor<1x8xf32>)
      -> tensor<1x?xf32>
      attributes {udon.onnx.operation = "BoundedPrefix",
                  udon.onnx.provider = "udononnxBoundedPrefix"}

  func.func @main_graph(
      %past: tensor<1x2x?x4xf32> {onnx.name = "past", udoncl.dim_exprs = [{axis = 2 : i64, offset = 0 : i64, symbol = "past"}]})
      -> (tensor<1x2x1x?x4xf32> {onnx.name = "viewed"},
          tensor<1x?xf32> {onnx.name = "prefix"}) {
    %axis = arith.constant dense<2> : tensor<1xi64>
    %viewed = func.call @__udon_onnx_unsqueeze(%past, %axis)
        {udon.onnx.result_dim_expr = {axis = 3 : i64, offset = 0 : i64, symbol = "past"}}
        : (tensor<1x2x?x4xf32>, tensor<1xi64>) -> tensor<1x2x1x?x4xf32>
    %bound = arith.constant dense<1.0> : tensor<1x8xf32>
    %prefix = func.call @__udon_onnx_boundedprefix(%bound)
        {udon.onnx.result_dim_expr = {axis = 1 : i64, offset = 1 : i64, symbol = "past"}}
        : (tensor<1x8xf32>) -> tensor<1x?xf32>
    return %viewed, %prefix : tensor<1x2x1x?x4xf32>, tensor<1x?xf32>
  }
}

// CHECK: module attributes {udoncl.dimension_bounds = {past = 7 : i64}, udoncl.onnx_host_graph_validated}
