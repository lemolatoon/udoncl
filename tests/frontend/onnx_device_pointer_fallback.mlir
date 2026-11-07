module {
  func.func @main_graph(
      %matrix: tensor<1x4xf32>, %weight: tensor<4x4xf32>,
      %left: tensor<2x2xi64>, %right: tensor<1xi64>)
      -> (tensor<1x4xf32>, tensor<2x2xi64>) {
    %product = "onnx.MatMul"(%matrix, %weight) :
        (tensor<1x4xf32>, tensor<4x4xf32>) -> tensor<1x4xf32>
    %sum = "onnx.Add"(%left, %right) :
        (tensor<2x2xi64>, tensor<1xi64>) -> tensor<2x2xi64>
    "onnx.Return"(%product, %sum) :
        (tensor<1x4xf32>, tensor<2x2xi64>) -> ()
  }
}
