module {
  func.func @main_graph(%left: tensor<2x2xi64>, %right: tensor<1xi64>)
      -> tensor<2x2xi64> {
    %result = "onnx.Add"(%left, %right) :
        (tensor<2x2xi64>, tensor<1xi64>) -> tensor<2x2xi64>
    "onnx.Return"(%result) : (tensor<2x2xi64>) -> ()
  }
}
