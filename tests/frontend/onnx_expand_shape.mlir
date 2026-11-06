module {
  func.func @expand(%input: tensor<1x32x1xf32>) -> tensor<*xf32> {
    %shape = "onnx.Constant"() {
      value = dense<1> : tensor<3xi64>
    } : () -> tensor<3xi64>
    %expanded = "onnx.Expand"(%input, %shape) :
        (tensor<1x32x1xf32>, tensor<3xi64>) -> tensor<*xf32>
    "onnx.Return"(%expanded) : (tensor<*xf32>) -> ()
  }
}
