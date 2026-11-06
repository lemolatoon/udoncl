module {
  func.func @main_graph(%left: tensor<2x1xf32>, %right: tensor<2x2xf32>)
      -> tensor<2x3xf32> {
    %result = "onnx.Concat"(%left, %right) {
      axis = 1 : si64
    } : (tensor<2x1xf32>, tensor<2x2xf32>) -> tensor<2x3xf32>
    "onnx.Return"(%result) : (tensor<2x3xf32>) -> ()
  }
}
