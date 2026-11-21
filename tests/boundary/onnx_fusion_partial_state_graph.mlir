// ONNX Host graph of the partial-advance part of the fused kernel execution
// test (onnx_fusion.py): the key cache of onnx_fusion_sequence_state_graph.mlir
// specialized to two new rows per run. Host finalization banks the state and
// synthesizes udononnx_fused_0, which reads the carried rows from the
// previous present at a run-time offset (the rows the last advance
// committed, at most two), and udononnx_fused_1, which reads the past input
// (bank 0). onnx_fusion.cpp commits one, two and no rows of successive runs
// and compares every present with a reference that carries exactly the
// committed rows.

module attributes {udoncl.onnx_sequence_states = [
    {axis = 2 : i64, capacity = 8 : i64, input_index = 1 : i64,
     output_index = 0 : i64, step = 2 : i64}]} {
  func.func private @clEnqueueMarkerWithWaitList(
      !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32

  func.func private @__udon_onnx_neg(tensor<1x2x2x8xf32>)
      -> tensor<1x2x2x8xf32>
      attributes {
        udon.onnx.operation = "Neg",
        udon.onnx.provider = "udononnxNeg"
      }
  func.func private @__udon_onnx_concat(
      tensor<1x2x6x8xf32>, tensor<1x2x2x8xf32>) -> tensor<1x2x8x8xf32>
      attributes {
        udon.onnx.operation = "Concat",
        udon.onnx.provider = "udononnxConcat"
      }

  func.func @main_graph(
      %token: tensor<1x2x2x8xf32> {onnx.name = "token"},
      %past_key: tensor<1x2x6x8xf32> {onnx.name = "past_key"})
      -> (tensor<1x2x8x8xf32> {onnx.name = "present_key"}) {
    %negated = func.call @__udon_onnx_neg(%token)
        : (tensor<1x2x2x8xf32>) -> tensor<1x2x2x8xf32>
    %present_key = func.call @__udon_onnx_concat(%past_key, %negated)
        {udon.onnx.attributes = {axis = 2 : si64}}
        : (tensor<1x2x6x8xf32>, tensor<1x2x2x8xf32>) -> tensor<1x2x8x8xf32>
    return %present_key : tensor<1x2x8x8xf32>
  }
}
