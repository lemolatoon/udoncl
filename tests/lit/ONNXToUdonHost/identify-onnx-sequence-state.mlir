// RUN: %vrc-opt --udoncl-identify-onnx-sequence-state %s | %FileCheck %s

// A Concat whose first operand is a non-empty graph input is append-only
// sequence state. A zero-extent input carries no rows into the next run, so
// it is an ordinary empty input rather than a fixed-capacity state bank.

module {
  func.func @main_graph(
      %token: tensor<1x4x1x64xf32>,
      %past_key: tensor<1x4x7x64xf32>,
      %empty_past: tensor<1x4x0x64xf32>,
      %prompt: tensor<1x4x4x64xf32>)
      -> (tensor<1x4x8x64xf32>, tensor<1x4x4x64xf32>) {
    %present_key = "onnx.Concat"(%past_key, %token) {axis = 2 : si64} :
        (tensor<1x4x7x64xf32>, tensor<1x4x1x64xf32>)
        -> tensor<1x4x8x64xf32>
    %present_prompt = "onnx.Concat"(%empty_past, %prompt) {axis = 2 : si64} :
        (tensor<1x4x0x64xf32>, tensor<1x4x4x64xf32>)
        -> tensor<1x4x4x64xf32>
    "onnx.Return"(%present_key, %present_prompt) :
        (tensor<1x4x8x64xf32>, tensor<1x4x4x64xf32>) -> ()
  }
}

// CHECK: module attributes {udoncl.onnx_sequence_states = [{axis = 2 : i64, capacity = 8 : i64, input_index = 1 : i64, output_index = 0 : i64, step = 1 : i64}]}
// CHECK-NOT: input_index = 2
