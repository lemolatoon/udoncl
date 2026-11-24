// RUN: not %vrc-opt --udoncl-validate-onnx-host-graph %s 2>&1 | %FileCheck %s

// A graph input whose dynamic axis is the inner one of an operand is not
// dense in a buffer of a larger pitch, which the kernel cannot address.
module attributes {udoncl.dimension_bounds = {past = 7 : i64}} {
  func.func private @__udon_onnx_matmul(tensor<2x1x?xf32>, tensor<2x?x3xf32>)
      -> tensor<2x1x3xf32>
      attributes {udon.onnx.operation = "MatMul",
                  udon.onnx.provider = "udonblasSgemm"}

  func.func @main_graph(
      %probabilities: tensor<2x1x?xf32> {onnx.name = "probabilities", udoncl.dim_exprs = [{axis = 2 : i64, offset = 0 : i64, symbol = "past"}]},
      %values: tensor<2x?x3xf32> {onnx.name = "values", udoncl.dim_exprs = [{axis = 1 : i64, offset = 0 : i64, symbol = "past"}]})
      -> (tensor<2x1x3xf32> {onnx.name = "attention"}) {
    %attention = func.call @__udon_onnx_matmul(%probabilities, %values)
        : (tensor<2x1x?xf32>, tensor<2x?x3xf32>) -> tensor<2x1x3xf32>
    return %attention : tensor<2x1x3xf32>
  }
}

// CHECK: error: ONNX operator MatMul does not support a dynamic extent on axis 2 (dimension past)
