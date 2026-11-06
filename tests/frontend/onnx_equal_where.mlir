module {
  func.func @main_graph(%input: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %sentinel = "onnx.Constant"() {
      value = dense<0.0> : tensor<f32>
    } : () -> tensor<f32>
    %replacement = "onnx.Constant"() {
      value = dense<1.0> : tensor<f32>
    } : () -> tensor<f32>
    %condition = "onnx.Equal"(%input, %sentinel) :
        (tensor<2x2xf32>, tensor<f32>) -> tensor<2x2xi1>
    %result = "onnx.Where"(%condition, %replacement, %input) :
        (tensor<2x2xi1>, tensor<f32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    "onnx.Return"(%result) : (tensor<2x2xf32>) -> ()
  }
}
