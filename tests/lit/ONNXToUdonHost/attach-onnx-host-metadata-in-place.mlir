// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph --udoncl-attach-onnx-host-metadata %s | %FileCheck %s
// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph --udoncl-attach-onnx-host-metadata --vrc-describe-onnx-binding-manifest %s | %FileCheck --check-prefix=MANIFEST %s

// A graph output that is Concat(graph input, x) along the axis of a bounded
// dimension, with the same element type and extents on the other axes and
// rows of whole texels, is an in-place Concat pair: a session may bind both
// to one buffer. Only `present` qualifies:
// - `reversed` takes the graph input as its second operand;
// - `narrow` has rows of 3 words, not a multiple of four;
// - `wide` concatenates a static axis.

module attributes {udoncl.dimension_bounds = {past = 7 : i64}} {
  func.func private @__udon_onnx_concat_0(tensor<1x2x?x4xf32>, tensor<1x2x1x4xf32>) -> tensor<1x2x?x4xf32>
      attributes {udon.onnx.operation = "Concat", udon.onnx.provider = "udononnxConcat"}
  func.func private @__udon_onnx_concat_1(tensor<1x2x1x4xf32>, tensor<1x2x?x4xf32>) -> tensor<1x2x?x4xf32>
      attributes {udon.onnx.operation = "Concat", udon.onnx.provider = "udononnxConcat"}
  func.func private @__udon_onnx_concat_2(tensor<1x?x3xf32>, tensor<1x1x3xf32>) -> tensor<1x?x3xf32>
      attributes {udon.onnx.operation = "Concat", udon.onnx.provider = "udononnxConcat"}
  func.func private @__udon_onnx_concat_3(tensor<1x?x4xf32>, tensor<1x?x4xf32>) -> tensor<2x?x4xf32>
      attributes {udon.onnx.operation = "Concat", udon.onnx.provider = "udononnxConcat"}

  func.func @main_graph(
      %past: tensor<1x2x?x4xf32> {onnx.name = "past", udoncl.dim_exprs = [{axis = 2 : i64, offset = 0 : i64, symbol = "past"}]},
      %step: tensor<1x2x1x4xf32> {onnx.name = "step"},
      %short: tensor<1x?x3xf32> {onnx.name = "short", udoncl.dim_exprs = [{axis = 1 : i64, offset = 0 : i64, symbol = "past"}]},
      %row: tensor<1x1x3xf32> {onnx.name = "row"},
      %rows: tensor<1x?x4xf32> {onnx.name = "rows", udoncl.dim_exprs = [{axis = 1 : i64, offset = 0 : i64, symbol = "past"}]})
      -> (tensor<1x2x?x4xf32> {onnx.name = "present"},
          tensor<1x2x?x4xf32> {onnx.name = "reversed"},
          tensor<1x?x3xf32> {onnx.name = "narrow"},
          tensor<2x?x4xf32> {onnx.name = "wide"}) {
    %present = func.call @__udon_onnx_concat_0(%past, %step)
        {udon.onnx.attributes = {axis = 2 : si64}, udon.onnx.result_dim_expr = {axis = 2 : i64, offset = 1 : i64, symbol = "past"}}
        : (tensor<1x2x?x4xf32>, tensor<1x2x1x4xf32>) -> tensor<1x2x?x4xf32>
    %reversed = func.call @__udon_onnx_concat_1(%step, %past)
        {udon.onnx.attributes = {axis = -2 : si64}, udon.onnx.result_dim_expr = {axis = 2 : i64, offset = 1 : i64, symbol = "past"}}
        : (tensor<1x2x1x4xf32>, tensor<1x2x?x4xf32>) -> tensor<1x2x?x4xf32>
    %narrow = func.call @__udon_onnx_concat_2(%short, %row)
        {udon.onnx.attributes = {axis = 1 : si64}, udon.onnx.result_dim_expr = {axis = 1 : i64, offset = 1 : i64, symbol = "past"}}
        : (tensor<1x?x3xf32>, tensor<1x1x3xf32>) -> tensor<1x?x3xf32>
    %wide = func.call @__udon_onnx_concat_3(%rows, %rows)
        {udon.onnx.attributes = {axis = 0 : si64}, udon.onnx.result_dim_expr = {axis = 1 : i64, offset = 0 : i64, symbol = "past"}}
        : (tensor<1x?x4xf32>, tensor<1x?x4xf32>) -> tensor<2x?x4xf32>
    return %present, %reversed, %narrow, %wide
        : tensor<1x2x?x4xf32>, tensor<1x2x?x4xf32>, tensor<1x?x3xf32>, tensor<2x?x4xf32>
  }
}

// CHECK: udoncl.onnx_binding = {abi = "udononnx-runtime-v3",
// CHECK-SAME: in_place_concats = [{axis = 2 : i32, input = 0 : i32, output = 0 : i32}]
// CHECK-SAME: model_index

// MANIFEST: vrc.onnx_binding_manifest = {abi = "udononnx-runtime-v3",
// MANIFEST-SAME: in_place_concats = [{axis = 2 : i64, input = 0 : i64, output = 0 : i64}]
