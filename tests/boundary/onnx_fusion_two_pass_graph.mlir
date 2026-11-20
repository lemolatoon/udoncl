// ONNX Host graph of the two-pass Softmax execution test (onnx_fusion.py):
//   long = softmax(x + mask) over the last axis of 32 rows of 512
// The row is long enough that Host finalization runs the Softmax root in two
// passes (deviceSoftmaxSchedule): a statistics kernel that writes chunk
// maxima and sums to the operator scratch, then an elementwise kernel.
// onnx_fusion.cpp compares the result with the library Softmax.

func.func private @clEnqueueMarkerWithWaitList(
    !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32

func.func private @__udon_onnx_add_row(
    tensor<1x32x1x512xf32>, tensor<1x1x1x512xf32>) -> tensor<1x32x1x512xf32>
    attributes {
      udon.onnx.operation = "Add",
      udon.onnx.provider = "udonatenAdd"
    }
func.func private @__udon_onnx_softmax(
    tensor<1x32x1x512xf32>) -> tensor<1x32x1x512xf32>
    attributes {
      udon.onnx.operation = "Softmax",
      udon.onnx.provider = "udondnnSoftmaxForward"
    }

func.func @main_graph(
    %x: tensor<1x32x1x512xf32> {onnx.name = "x"},
    %mask: tensor<1x1x1x512xf32> {onnx.name = "mask"})
    -> (tensor<1x32x1x512xf32> {onnx.name = "long"}) {
  %scores = func.call @__udon_onnx_add_row(%x, %mask)
      : (tensor<1x32x1x512xf32>, tensor<1x1x1x512xf32>)
      -> tensor<1x32x1x512xf32>
  %long = func.call @__udon_onnx_softmax(%scores)
      {udon.onnx.attributes = {axis = -1 : si64}}
      : (tensor<1x32x1x512xf32>) -> tensor<1x32x1x512xf32>
  return %long : tensor<1x32x1x512xf32>
}
