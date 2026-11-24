// RUN: %vrc-opt --udoncl-simplify-onnx-empty-concat %s | %FileCheck %s

// An operand with a dynamic extent of a bounded dimension may be empty at
// some extents only (a past of length 0), so it is never dropped.

module {
  func.func @main_graph(%past: tensor<1x2x?x4xf32>, %new: tensor<1x2x1x4xf32>)
      -> tensor<1x2x?x4xf32> {
    %present = "onnx.Concat"(%past, %new) {axis = 2 : si64}
        : (tensor<1x2x?x4xf32>, tensor<1x2x1x4xf32>) -> tensor<1x2x?x4xf32>
    "onnx.Return"(%present) : (tensor<1x2x?x4xf32>) -> ()
  }
}

// CHECK: %[[PRESENT:.*]] = "onnx.Concat"(%arg0, %arg1)
// CHECK: onnx.Return %[[PRESENT]]
