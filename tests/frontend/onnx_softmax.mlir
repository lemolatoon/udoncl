module {
  func.func @main_graph(%input: tensor<1x2x3xf32>) -> tensor<1x2x3xf32> {
    %result = "onnx.Softmax"(%input) {
      axis = -1 : si64
    } : (tensor<1x2x3xf32>) -> tensor<1x2x3xf32>
    "onnx.Return"(%result) : (tensor<1x2x3xf32>) -> ()
  }
}
