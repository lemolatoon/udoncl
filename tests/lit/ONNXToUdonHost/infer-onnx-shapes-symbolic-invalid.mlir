// RUN: not %vrc-opt --split-input-file --udoncl-infer-onnx-shapes %s 2>&1 | %FileCheck %s

// A product of bounded dimensions is not an affine extent.
module attributes {udoncl.dimension_bounds = {past = 8 : i64}} {
  func.func @main_graph(
      %past: tensor<1x?xf32> {udoncl.dim_exprs = [{axis = 1 : i64, offset = 0 : i64, symbol = "past"}]})
      -> tensor<*xf32> {
    %shape = "onnx.Shape"(%past) : (tensor<1x?xf32>) -> tensor<2xi64>
    %one = onnx.Constant dense<1> : tensor<1xi64>
    %length = "onnx.Gather"(%shape, %one) {axis = 0 : si64}
        : (tensor<2xi64>, tensor<1xi64>) -> tensor<*xi64>
    %square = "onnx.Mul"(%length, %length)
        : (tensor<*xi64>, tensor<*xi64>) -> tensor<*xi64>
    %filled = onnx.ConstantOfShape(%square) {value = dense<0.0> : tensor<1xf32>}
        : (tensor<*xi64>) -> tensor<*xf32>
    "onnx.Return"(%filled) : (tensor<*xf32>) -> ()
  }
}

// CHECK: error: ONNX operation computes a product of bounded dimensions (past) * (past)

// -----

// Every tensor has at most one bounded dynamic axis.
module attributes {udoncl.dimension_bounds = {past = 8 : i64}} {
  func.func @main_graph(
      %past: tensor<1x?xf32> {udoncl.dim_exprs = [{axis = 1 : i64, offset = 0 : i64, symbol = "past"}]})
      -> tensor<*xf32> {
    %shape = "onnx.Shape"(%past) : (tensor<1x?xf32>) -> tensor<2xi64>
    %one = onnx.Constant dense<1> : tensor<1xi64>
    %length = "onnx.Gather"(%shape, %one) {axis = 0 : si64}
        : (tensor<2xi64>, tensor<1xi64>) -> tensor<*xi64>
    %square = "onnx.Concat"(%length, %length) {axis = 0 : si64}
        : (tensor<*xi64>, tensor<*xi64>) -> tensor<*xi64>
    %filled = onnx.ConstantOfShape(%square) {value = dense<0.0> : tensor<1xf32>}
        : (tensor<*xi64>) -> tensor<*xf32>
    "onnx.Return"(%filled) : (tensor<*xf32>) -> ()
  }
}

// CHECK: error: ONNX operation has more than one bounded dynamic axis: [past, past]
