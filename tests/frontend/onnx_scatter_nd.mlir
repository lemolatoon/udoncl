module {
  func.func @main_graph(%data: tensor<4xf32>, %updates: tensor<2xf32>)
      -> tensor<4xf32> {
    %indices = "onnx.Constant"() {
      value = dense<[[1], [3]]> : tensor<2x1xi64>
    } : () -> tensor<2x1xi64>
    %result = "onnx.ScatterND"(%data, %indices, %updates) {
      reduction = "none"
    } : (tensor<4xf32>, tensor<2x1xi64>, tensor<2xf32>) -> tensor<4xf32>
    "onnx.Return"(%result) : (tensor<4xf32>) -> ()
  }
}
