// RUN: %vrc-opt --udoncl-infer-onnx-shapes %s | %FileCheck %s

// With bounded dimensions, shape values are evaluated as constants or
// `symbol + offset`: an axis that depends on a bounded dimension stays
// dynamic and every other extent becomes static, as in the Hugging Face
// export of a decoder with a bounded past.

module attributes {udoncl.dimension_bounds = {past = 8 : i64}} {
  func.func @main_graph(
      %past: tensor<1x2x?x4xf32> {udoncl.dim_exprs = [{axis = 2 : i64, offset = 0 : i64, symbol = "past"}]},
      %new: tensor<1x2x1x4xf32>,
      %mask: tensor<1x?xi64> {udoncl.dim_exprs = [{axis = 1 : i64, offset = 1 : i64, symbol = "past"}]})
      -> (tensor<*xf32>, tensor<*xf32>, tensor<*xi64>, tensor<*xf32>, tensor<*xi64>) {
    %present = "onnx.Concat"(%past, %new) {axis = -2 : si64}
        : (tensor<1x2x?x4xf32>, tensor<1x2x1x4xf32>) -> tensor<*xf32>
    // Shape -> Gather -> Concat -> Reshape: the merged heads keep the
    // dynamic axis and the static extents.
    %shape = "onnx.Shape"(%present) : (tensor<*xf32>) -> tensor<4xi64>
    %two = onnx.Constant dense<2> : tensor<1xi64>
    %length = "onnx.Gather"(%shape, %two) {axis = 0 : si64}
        : (tensor<4xi64>, tensor<1xi64>) -> tensor<*xi64>
    %one = onnx.Constant dense<1> : tensor<1xi64>
    %four = onnx.Constant dense<4> : tensor<1xi64>
    %eight = onnx.Constant dense<8> : tensor<1xi64>
    %heads = "onnx.Concat"(%one, %length, %eight) {axis = 0 : si64}
        : (tensor<1xi64>, tensor<*xi64>, tensor<1xi64>) -> tensor<*xi64>
    %transposed = "onnx.Transpose"(%present) {perm = [0, 2, 1, 3]}
        : (tensor<*xf32>) -> tensor<*xf32>
    %merged = "onnx.Reshape"(%transposed, %heads)
        : (tensor<*xf32>, tensor<*xi64>) -> tensor<*xf32>
    // Range from 0 to the mask length: its extent is past + 1.
    %mask_shape = "onnx.Shape"(%mask) : (tensor<1x?xi64>) -> tensor<2xi64>
    %mask_length = "onnx.Gather"(%mask_shape, %one) {axis = 0 : si64}
        : (tensor<2xi64>, tensor<1xi64>) -> tensor<*xi64>
    %zero = onnx.Constant dense<0> : tensor<i64>
    %step = onnx.Constant dense<1> : tensor<i64>
    %zero_axis = onnx.Constant dense<0> : tensor<1xi64>
    %limit = "onnx.Squeeze"(%mask_length, %zero_axis)
        : (tensor<*xi64>, tensor<1xi64>) -> tensor<*xi64>
    %columns = "onnx.Range"(%zero, %limit, %step)
        : (tensor<i64>, tensor<*xi64>, tensor<i64>) -> tensor<*xi64>
    // Expand to [1, 2, 3, past + 1, 4] keeps the past axis dynamic.
    %axis = onnx.Constant dense<2> : tensor<1xi64>
    %unsqueezed = "onnx.Unsqueeze"(%present, %axis)
        : (tensor<*xf32>, tensor<1xi64>) -> tensor<*xf32>
    %leading = onnx.Constant dense<[1, 2, 3]> : tensor<3xi64>
    %extent = "onnx.Concat"(%leading, %length, %four) {axis = 0 : si64}
        : (tensor<3xi64>, tensor<*xi64>, tensor<1xi64>) -> tensor<*xi64>
    %repeated = "onnx.Expand"(%unsqueezed, %extent)
        : (tensor<*xf32>, tensor<*xi64>) -> tensor<*xf32>
    // past + 1 - 1 == past: a shape value that only depends on the bounded
    // dimension through a difference of equal terms folds to a constant.
    %past_shape = "onnx.Shape"(%past) : (tensor<1x2x?x4xf32>) -> tensor<4xi64>
    %past_length = "onnx.Gather"(%past_shape, %two) {axis = 0 : si64}
        : (tensor<4xi64>, tensor<1xi64>) -> tensor<*xi64>
    %difference = "onnx.Sub"(%length, %past_length)
        : (tensor<*xi64>, tensor<*xi64>) -> tensor<*xi64>
    "onnx.Return"(%present, %merged, %columns, %repeated, %difference)
        : (tensor<*xf32>, tensor<*xf32>, tensor<*xi64>, tensor<*xf32>, tensor<*xi64>) -> ()
  }
}

// CHECK-LABEL: func.func @main_graph
// CHECK-SAME: -> (tensor<1x2x?x4xf32>, tensor<1x?x8xf32>, tensor<?xi64>, tensor<1x2x3x?x4xf32>, tensor<1xi64>)
// CHECK: "onnx.Concat"(%arg0, %arg1) {{.*}} -> tensor<1x2x?x4xf32>
// CHECK: "onnx.Reshape"({{.*}}) {{.*}}: (tensor<1x?x2x4xf32>, tensor<3xi64>) -> tensor<1x?x8xf32>
// CHECK: "onnx.Range"({{.*}}) -> tensor<?xi64>
// CHECK: "onnx.Expand"({{.*}}) : (tensor<1x2x1x?x4xf32>, tensor<5xi64>) -> tensor<1x2x3x?x4xf32>
// CHECK-NOT: "onnx.Sub"
// CHECK: %[[DIFFERENCE:.*]] = arith.constant dense<1> : tensor<1xi64>
// CHECK: onnx.Return {{.*}}, %[[DIFFERENCE]]
