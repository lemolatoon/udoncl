module {
  func.func @main_graph(%input: tensor<1x256xf32>) -> tensor<1x128xf32> {
    %weight = "onnx.Constant"() {
      value = dense<0.5> : tensor<256x128xf32>
    } : () -> tensor<256x128xf32>
    %result = "onnx.MatMul"(%input, %weight) :
        (tensor<1x256xf32>, tensor<256x128xf32>) -> tensor<1x128xf32>
    "onnx.Return"(%result) : (tensor<1x128xf32>) -> ()
  }
}
