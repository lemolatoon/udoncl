module {
  func.func @main_graph(%input: tensor<1x1024xf32>) -> tensor<1x8xf32> {
    %weight = "onnx.Constant"() {
      value = dense<0.5> : tensor<1024x8xf32>
    } : () -> tensor<1024x8xf32>
    %result = "onnx.MatMul"(%input, %weight) :
        (tensor<1x1024xf32>, tensor<1024x8xf32>) -> tensor<1x8xf32>
    "onnx.Return"(%result) : (tensor<1x8xf32>) -> ()
  }
}
