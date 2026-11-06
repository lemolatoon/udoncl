module {
  func.func @main_graph(%input: tensor<1x20000xf32>) -> tensor<1x20000xf32> {
    %product = "onnx.Mul"(%input, %input) :
        (tensor<1x20000xf32>, tensor<1x20000xf32>) -> tensor<1x20000xf32>
    %sigmoid = "onnx.Sigmoid"(%product) :
        (tensor<1x20000xf32>) -> tensor<1x20000xf32>
    %tanh = "onnx.Tanh"(%sigmoid) :
        (tensor<1x20000xf32>) -> tensor<1x20000xf32>
    "onnx.Return"(%tanh) : (tensor<1x20000xf32>) -> ()
  }
}
