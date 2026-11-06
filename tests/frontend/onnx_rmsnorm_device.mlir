#weight = loc("Initializer_weight")
#epsilon = loc("Initializer_epsilon")
module {
  func.func @main_graph(%input: tensor<4x2048xf32> {onnx.name = "input"})
      -> tensor<4x2048xf32> {
    %weight = "onnx.Constant"() {
      value = dense<1.0> : tensor<2048xf32>
    } : () -> tensor<2048xf32> loc(#weight)
    %epsilon = "onnx.Constant"() {
      value = dense<1.0e-5> : tensor<f32>
    } : () -> tensor<f32> loc(#epsilon)
    %output = "onnx.Custom"(%input, %weight, %epsilon)
        <{function_name = "RmsNorm"}>
        {domain_name = "com.lemolatoon.udoncl"} :
        (tensor<4x2048xf32>, tensor<2048xf32>, tensor<f32>)
        -> tensor<4x2048xf32>
    "onnx.Return"(%output) : (tensor<4x2048xf32>) -> ()
  }
}
