module {
  func.func @main_graph(%input: tensor<2x2x3xf32>) -> tensor<2x2x4xf32> {
    %weight = "onnx.Constant"() {
      value = dense<1.0> : tensor<1x3x4xf32>
    } : () -> tensor<1x3x4xf32>
    %result = "onnx.MatMul"(%input, %weight) :
        (tensor<2x2x3xf32>, tensor<1x3x4xf32>) -> tensor<2x2x4xf32>
    "onnx.Return"(%result) : (tensor<2x2x4xf32>) -> ()
  }
}
