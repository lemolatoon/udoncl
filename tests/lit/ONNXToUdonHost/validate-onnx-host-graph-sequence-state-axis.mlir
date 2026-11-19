// RUN: %vrc-opt --udoncl-validate-onnx-host-graph %s | %FileCheck %s

// Sequence-state metadata records a normalized axis. A Concat that spells the
// same axis as a negative ONNX axis still matches it.

module attributes {udoncl.onnx_sequence_states = [{axis = 2 : i64, capacity = 4 : i64, input_index = 0 : i64, output_index = 0 : i64, step = 1 : i64}]} {
  func.func private @__udon_onnx_concat(
      tensor<1x2x3x8xf32>, tensor<1x2x1x8xf32>) -> tensor<1x2x4x8xf32>
      attributes {
        udon.onnx.operation = "Concat",
        udon.onnx.provider = "udononnxConcat"
      }

  func.func @main_graph(
      %past: tensor<1x2x3x8xf32> {onnx.name = "past"},
      %step: tensor<1x2x1x8xf32> {onnx.name = "step"})
      -> (tensor<1x2x4x8xf32> {onnx.name = "present"}) {
    %present = func.call @__udon_onnx_concat(%past, %step)
        {udon.onnx.attributes = {axis = -2 : si64}}
        : (tensor<1x2x3x8xf32>, tensor<1x2x1x8xf32>) -> tensor<1x2x4x8xf32>
    return %present : tensor<1x2x4x8xf32>
  }
}

// CHECK: udoncl.onnx_host_graph_validated
// CHECK: func.func @main_graph
