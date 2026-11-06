#initializer = loc("Initializer_weight")
module {
  func.func @main_graph(%input: tensor<1xf32> {onnx.name = "input"}) -> tensor<2x2xf32> {
    %weight = "onnx.Constant"() {value = dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>} : () -> tensor<2x2xf32> loc(#initializer)
    "onnx.Return"(%weight) : (tensor<2x2xf32>) -> ()
  }
}
