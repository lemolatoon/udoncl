// RUN: %vrc-opt --udoncl-lower-onnx-to-host %s | %FileCheck %s

// A provider result with an axis of a bounded dimension records its extent
// expression, and a bounded prefix lowers to the BoundedPrefix view.

module attributes {udoncl.dimension_bounds = {past = 3 : i64}} {
  func.func @main_graph(
      %past: tensor<1x?xf32> {onnx.name = "past", udoncl.dim_exprs = [{axis = 1 : i64, offset = 0 : i64, symbol = "past"}]},
      %new: tensor<1x1xf32> {onnx.name = "new"})
      -> (tensor<1x?xf32> {onnx.dim_params = "1:past + 1", onnx.name = "present"},
          tensor<1x?xf32> {onnx.dim_params = "1:past + 1", onnx.name = "mask"}) {
    %present = "onnx.Concat"(%past, %new) {axis = 1 : si64}
        : (tensor<1x?xf32>, tensor<1x1xf32>) -> tensor<1x?xf32>
    %bound = arith.constant dense<0.0> : tensor<1x4xf32>
    %mask = "onnx.Custom"(%bound) {function_name = "UdonclBoundedPrefix", domain_name = "com.lemolatoon.udoncl", axis = 1 : i64, symbol = "past", offset = 1 : i64}
        : (tensor<1x4xf32>) -> tensor<1x?xf32>
    "onnx.Return"(%present, %mask) : (tensor<1x?xf32>, tensor<1x?xf32>) -> ()
  }
}

// CHECK-DAG: func.func private @{{.*}}(tensor<1x4xf32>) -> tensor<1x?xf32> attributes {udon.onnx.domain = "com.lemolatoon.udoncl", udon.onnx.operation = "BoundedPrefix", udon.onnx.provider = "udononnxBoundedPrefix"}
// CHECK-DAG: func.func private @{{.*}}(tensor<1x?xf32>, tensor<1x1xf32>) -> tensor<1x?xf32> attributes {udon.onnx.operation = "Concat", udon.onnx.provider = "udononnxConcat"}
// CHECK: call @{{.*}}(%arg0, %arg1) {udon.onnx.attributes = {axis = 1 : si64}, udon.onnx.result_dim_expr = {axis = 1 : i64, offset = 1 : i64, symbol = "past"}}
// CHECK: call @{{.*}}(%{{.*}}) {udon.onnx.attributes = {{{.*}}}, udon.onnx.result_dim_expr = {axis = 1 : i64, offset = 1 : i64, symbol = "past"}}
