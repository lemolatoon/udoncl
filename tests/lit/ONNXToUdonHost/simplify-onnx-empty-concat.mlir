// RUN: %vrc-opt --udoncl-simplify-onnx-empty-concat %s | %FileCheck %s

// Statically empty Concat operands are removed. A single remaining operand of
// the result type replaces the Concat; otherwise the Concat keeps the
// remaining operands. An all-empty Concat is left unchanged.

module {
  func.func @main_graph(
      %empty: tensor<1x4x0x64xf32>,
      %current: tensor<1x4x4x64xf32>,
      %more: tensor<1x4x2x64xf32>,
      %other_empty: tensor<1x4x0x64xf32>)
      -> (tensor<1x4x4x64xf32>, tensor<1x4x6x64xf32>, tensor<1x4x0x64xf32>) {
    %only = "onnx.Concat"(%empty, %current) {axis = 2 : si64}
        : (tensor<1x4x0x64xf32>, tensor<1x4x4x64xf32>) -> tensor<1x4x4x64xf32>
    %pair = "onnx.Concat"(%current, %empty, %more) {axis = 2 : si64}
        : (tensor<1x4x4x64xf32>, tensor<1x4x0x64xf32>, tensor<1x4x2x64xf32>)
        -> tensor<1x4x6x64xf32>
    %none = "onnx.Concat"(%empty, %other_empty) {axis = 2 : si64}
        : (tensor<1x4x0x64xf32>, tensor<1x4x0x64xf32>) -> tensor<1x4x0x64xf32>
    "onnx.Return"(%only, %pair, %none)
        : (tensor<1x4x4x64xf32>, tensor<1x4x6x64xf32>, tensor<1x4x0x64xf32>)
        -> ()
  }
}

// CHECK-LABEL: func.func @main_graph
// CHECK-SAME: (%[[EMPTY:.*]]: tensor<1x4x0x64xf32>, %[[CURRENT:.*]]: tensor<1x4x4x64xf32>, %[[MORE:.*]]: tensor<1x4x2x64xf32>, %[[OTHER:.*]]: tensor<1x4x0x64xf32>)
// CHECK: %[[PAIR:.*]] = "onnx.Concat"(%[[CURRENT]], %[[MORE]]) <{axis = 2 : si64}>
// CHECK: %[[NONE:.*]] = "onnx.Concat"(%[[EMPTY]], %[[OTHER]]) <{axis = 2 : si64}>
// CHECK: onnx.Return %[[CURRENT]], %[[PAIR]], %[[NONE]]
