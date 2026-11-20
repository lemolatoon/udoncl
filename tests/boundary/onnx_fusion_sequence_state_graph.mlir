// ONNX Host graph of the banked sequence-state part of the fused kernel
// execution test (onnx_fusion.py): a key cache of two heads, capacity 8,
// that appends the negated token row. Host finalization banks the state and
// synthesizes two kernels: udononnx_fused_0 reads the carried rows from the
// previous present (banks 1 and 2), udononnx_fused_1 from the past input
// (bank 0). onnx_fusion.cpp alternates the banks over several steps and
// compares every present with the copying advance built from the library
// Neg kernel.

module attributes {udoncl.onnx_sequence_states = [
    {axis = 2 : i64, capacity = 8 : i64, input_index = 1 : i64,
     output_index = 0 : i64, step = 1 : i64}]} {
  func.func private @clEnqueueMarkerWithWaitList(
      !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32

  func.func private @__udon_onnx_neg(tensor<1x2x1x8xf32>)
      -> tensor<1x2x1x8xf32>
      attributes {
        udon.onnx.operation = "Neg",
        udon.onnx.provider = "udononnxNeg"
      }
  func.func private @__udon_onnx_concat(
      tensor<1x2x7x8xf32>, tensor<1x2x1x8xf32>) -> tensor<1x2x8x8xf32>
      attributes {
        udon.onnx.operation = "Concat",
        udon.onnx.provider = "udononnxConcat"
      }

  func.func @main_graph(
      %token: tensor<1x2x1x8xf32> {onnx.name = "token"},
      %past_key: tensor<1x2x7x8xf32> {onnx.name = "past_key"})
      -> (tensor<1x2x8x8xf32> {onnx.name = "present_key"}) {
    %negated = func.call @__udon_onnx_neg(%token)
        : (tensor<1x2x1x8xf32>) -> tensor<1x2x1x8xf32>
    %present_key = func.call @__udon_onnx_concat(%past_key, %negated)
        {udon.onnx.attributes = {axis = 2 : si64}}
        : (tensor<1x2x7x8xf32>, tensor<1x2x1x8xf32>) -> tensor<1x2x8x8xf32>
    return %present_key : tensor<1x2x8x8xf32>
  }
}
