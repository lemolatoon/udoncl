// RUN: %vrc-opt --udoncl-bound-onnx-shape-operands %s | %FileCheck %s

// Shape operands of run-time operators that depend on the bounded `past`
// become extent-free, and the causal mask computed from shapes becomes a
// bounded prefix of its value at the bound (past = 3, mask length 4).

module attributes {udoncl.dimension_bounds = {past = 3 : i64}} {
  func.func @main_graph(
      %query: tensor<1x4x1x2xf32>,
      %key: tensor<1x2x1x2xf32>,
      %mask: tensor<1x?xi64> {udoncl.dim_exprs = [{axis = 1 : i64, offset = 1 : i64, symbol = "past"}]},
      %past: tensor<1x2x?x2xf32> {udoncl.dim_exprs = [{axis = 2 : i64, offset = 0 : i64, symbol = "past"}]})
      -> (tensor<1x4x1x?xf32>, tensor<1x2x?x2xf32>) {
    %present = "onnx.Concat"(%past, %key) {axis = 2 : si64}
        : (tensor<1x2x?x2xf32>, tensor<1x2x1x2xf32>) -> tensor<1x2x?x2xf32>
    // The causal row of the single query: columns 0..past, none in the
    // future, so Mul(Trilu(min), 0) is 0 then -0.
    %zero = onnx.Constant dense<0> : tensor<i64>
    %one = onnx.Constant dense<1> : tensor<i64>
    %one_vector = onnx.Constant dense<1> : tensor<1xi64>
    %past_length_vector = "onnx.Dim"(%past) {axis = 2 : si64}
        : (tensor<1x2x?x2xf32>) -> tensor<1xi64>
    %mask_length_vector = "onnx.Dim"(%mask) {axis = 1 : si64}
        : (tensor<1x?xi64>) -> tensor<1xi64>
    %squeeze_axis = onnx.Constant dense<0> : tensor<1xi64>
    %past_length = "onnx.Squeeze"(%past_length_vector, %squeeze_axis)
        : (tensor<1xi64>, tensor<1xi64>) -> tensor<i64>
    %mask_length = "onnx.Squeeze"(%mask_length_vector, %squeeze_axis)
        : (tensor<1xi64>, tensor<1xi64>) -> tensor<i64>
    %columns = "onnx.Range"(%zero, %mask_length, %one)
        : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<?xi64>
    %row_end = "onnx.Add"(%past_length, %one) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %rows = "onnx.Range"(%past_length, %row_end, %one)
        : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1xi64>
    %extent = "onnx.Concat"(%one_vector, %mask_length_vector) {axis = 0 : si64}
        : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
    %full = "onnx.ConstantOfShape"(%extent) {value = dense<-3.40282347E+38> : tensor<1xf32>}
        : (tensor<2xi64>) -> tensor<1x?xf32>
    %upper = "onnx.Trilu"(%full, %one) {upper = 1 : si64}
        : (tensor<1x?xf32>, tensor<i64>) -> tensor<1x?xf32>
    %column_shape = onnx.Constant dense<[-1, 1]> : tensor<2xi64>
    %row_column = "onnx.Reshape"(%rows, %column_shape)
        : (tensor<1xi64>, tensor<2xi64>) -> tensor<1x1xi64>
    %future = "onnx.Greater"(%columns, %row_column)
        : (tensor<?xi64>, tensor<1x1xi64>) -> tensor<1x?xi1>
    %future_f = "onnx.Cast"(%future) {saturate = 1 : si64, to = f32}
        : (tensor<1x?xi1>) -> tensor<1x?xf32>
    %causal = "onnx.Mul"(%upper, %future_f)
        : (tensor<1x?xf32>, tensor<1x?xf32>) -> tensor<1x?xf32>
    %axes = onnx.Constant dense<[0, 1]> : tensor<2xi64>
    %causal4 = "onnx.Unsqueeze"(%causal, %axes)
        : (tensor<1x?xf32>, tensor<2xi64>) -> tensor<1x1x1x?xf32>
    // repeat_kv: Expand to the present length and Reshape to the heads.
    %two = onnx.Constant dense<2> : tensor<1xi64>
    %unsqueezed = "onnx.Unsqueeze"(%present, %two)
        : (tensor<1x2x?x2xf32>, tensor<1xi64>) -> tensor<1x2x1x?x2xf32>
    %present_length = "onnx.Dim"(%present) {axis = 2 : si64}
        : (tensor<1x2x?x2xf32>) -> tensor<1xi64>
    %leading = onnx.Constant dense<[1, 2, 2]> : tensor<3xi64>
    %head_dim = onnx.Constant dense<2> : tensor<1xi64>
    %repeat = "onnx.Concat"(%leading, %present_length, %head_dim) {axis = 0 : si64}
        : (tensor<3xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<5xi64>
    %repeated = "onnx.Expand"(%unsqueezed, %repeat)
        : (tensor<1x2x1x?x2xf32>, tensor<5xi64>) -> tensor<1x2x2x?x2xf32>
    %heads_prefix = onnx.Constant dense<[1, 4]> : tensor<2xi64>
    %heads_shape = "onnx.Concat"(%heads_prefix, %present_length, %head_dim) {axis = 0 : si64}
        : (tensor<2xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
    %heads = "onnx.Reshape"(%repeated, %heads_shape)
        : (tensor<1x2x2x?x2xf32>, tensor<4xi64>) -> tensor<1x4x?x2xf32>
    %transposed = "onnx.Transpose"(%heads) {perm = [0, 1, 3, 2]}
        : (tensor<1x4x?x2xf32>) -> tensor<1x4x2x?xf32>
    %scores = "onnx.MatMul"(%query, %transposed)
        : (tensor<1x4x1x2xf32>, tensor<1x4x2x?xf32>) -> tensor<1x4x1x?xf32>
    // The mask sliced to the present length is the whole mask.
    %start = onnx.Constant dense<0> : tensor<1xi64>
    %axis = onnx.Constant dense<3> : tensor<1xi64>
    %score_mask = "onnx.Slice"(%causal4, %start, %present_length, %axis, %one_vector)
        : (tensor<1x1x1x?xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x1x1x?xf32>
    %masked = "onnx.Add"(%scores, %score_mask)
        : (tensor<1x4x1x?xf32>, tensor<1x1x1x?xf32>) -> tensor<1x4x1x?xf32>
    "onnx.Return"(%masked, %present) : (tensor<1x4x1x?xf32>, tensor<1x2x?x2xf32>) -> ()
  }
}

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: onnx.Dim
// CHECK-NOT: onnx.Range
// CHECK-NOT: onnx.ConstantOfShape
// CHECK-NOT: onnx.Trilu
// CHECK: %[[REPEAT:.*]] = arith.constant dense<[1, 2, 2, 1, 2]> : tensor<5xi64>
// CHECK: "onnx.Expand"(%{{.*}}, %[[REPEAT]])
// CHECK: %[[HEADS:.*]] = arith.constant dense<[1, 4, -1, 2]> : tensor<4xi64>
// CHECK: "onnx.Reshape"(%{{.*}}, %[[HEADS]])
// CHECK: %[[SCORES:.*]] = "onnx.MatMul"
// CHECK: %[[PREFIX:.*]] = arith.constant dense<{{\[\[\[\[}}0.000000e+00, -0.000000e+00, -0.000000e+00, -0.000000e+00]]]]> : tensor<1x1x1x4xf32>
// CHECK: %[[MASK:.*]] = "onnx.Custom"(%[[PREFIX]]) <{function_name = "UdonclBoundedPrefix"}> {axis = 3 : i64, domain_name = "com.lemolatoon.udoncl", offset = 1 : i64, symbol = "past"} : (tensor<1x1x1x4xf32>) -> tensor<1x1x1x?xf32>
// CHECK: "onnx.Add"(%[[SCORES]], %[[MASK]])
// CHECK-NOT: onnx.Slice
