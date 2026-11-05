module {
  func.func @main_graph(%input: tensor<1x4xf32>) -> tensor<1x4xf32> {
    %result = "onnx.Log"(%input) : (tensor<1x4xf32>) -> tensor<1x4xf32>
    return %result : tensor<1x4xf32>
  }
}
