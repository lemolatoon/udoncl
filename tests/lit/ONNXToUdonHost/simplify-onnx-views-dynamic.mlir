// RUN: %vrc-opt --udoncl-simplify-onnx-views %s | %FileCheck %s

// With a dynamic extent (a bounded dimension), equal types do not prove a
// view is the identity: a Slice must run from 0 to INT64_MAX on the dynamic
// axis and an Expand must take 1 there.

module {
  func.func @main_graph(%x: tensor<1x1x1x?xf32>, %y: tensor<1x4x?x2xf32>,
                        %z: tensor<1x1x?x2xf32>)
      -> (tensor<1x1x1x?xf32>, tensor<1x1x1x?xf32>, tensor<1x1x1x?xf32>,
          tensor<1x4x?x2xf32>, tensor<1x?x4x2xf32>, tensor<1x?x1x2xf32>) {
    %zero = onnx.Constant dense<0> : tensor<1xi64>
    %one = onnx.Constant dense<1> : tensor<1xi64>
    %max = onnx.Constant dense<9223372036854775807> : tensor<1xi64>
    %axis = onnx.Constant dense<3> : tensor<1xi64>
    // Removed: the whole dynamic extent.
    %whole = "onnx.Slice"(%x, %zero, %max, %axis, %one)
        : (tensor<1x1x1x?xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x1x1x?xf32>
    // Kept: drops the first element at every extent.
    %tail = "onnx.Slice"(%x, %one, %max, %axis, %one)
        : (tensor<1x1x1x?xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x1x1x?xf32>
    // Kept: an end that is not INT64_MAX may cut the extent.
    %eight = onnx.Constant dense<8> : tensor<1xi64>
    %head = "onnx.Slice"(%x, %zero, %eight, %axis, %one)
        : (tensor<1x1x1x?xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x1x1x?xf32>
    // Removed: an Expand that keeps the dynamic extent.
    %keep = onnx.Constant dense<[1, 4, 1, 2]> : tensor<4xi64>
    %same = "onnx.Expand"(%y, %keep) : (tensor<1x4x?x2xf32>, tensor<4xi64>) -> tensor<1x4x?x2xf32>
    // Kept: the dynamic axis moves across a non-unit axis.
    %moved = "onnx.Transpose"(%y) {perm = [0, 2, 1, 3]}
        : (tensor<1x4x?x2xf32>) -> tensor<1x?x4x2xf32>
    // A Transpose that moves the dynamic axis across unit axes only is a
    // Reshape whose dynamic extent is the inferred one.
    %flat = "onnx.Transpose"(%z) {perm = [0, 2, 1, 3]}
        : (tensor<1x1x?x2xf32>) -> tensor<1x?x1x2xf32>
    "onnx.Return"(%whole, %tail, %head, %same, %moved, %flat)
        : (tensor<1x1x1x?xf32>, tensor<1x1x1x?xf32>, tensor<1x1x1x?xf32>,
           tensor<1x4x?x2xf32>, tensor<1x?x4x2xf32>, tensor<1x?x1x2xf32>) -> ()
  }
}

// CHECK-LABEL: func.func @main_graph
// CHECK-DAG: %[[SHAPE:.*]] = arith.constant dense<[1, -1, 1, 2]> : tensor<4xi64>
// CHECK-DAG: %[[TAIL:.*]] = "onnx.Slice"(%arg0,
// CHECK-DAG: %[[HEAD:.*]] = "onnx.Slice"(%arg0,
// CHECK-DAG: %[[MOVED:.*]] = "onnx.Transpose"(%arg1)
// CHECK-DAG: %[[FLAT:.*]] = "onnx.Reshape"(%arg2, %[[SHAPE]])
// CHECK-NOT: onnx.Expand
// CHECK: onnx.Return %arg0, %[[TAIL]], %[[HEAD]], %arg1, %[[MOVED]], %[[FLAT]]
