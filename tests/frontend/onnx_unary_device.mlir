module {
  func.func @main_graph(%input: tensor<1x4xf32>) -> tensor<1x4xf32> {
    %product = "onnx.Mul"(%input, %input) :
        (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
    %cosine = "onnx.Cos"(%product) :
        (tensor<1x4xf32>) -> tensor<1x4xf32>
    %sine = "onnx.Sin"(%cosine) :
        (tensor<1x4xf32>) -> tensor<1x4xf32>
    %negative = "onnx.Neg"(%sine) :
        (tensor<1x4xf32>) -> tensor<1x4xf32>
    %sigmoid = "onnx.Sigmoid"(%negative) :
        (tensor<1x4xf32>) -> tensor<1x4xf32>
    %tanh = "onnx.Tanh"(%sigmoid) :
        (tensor<1x4xf32>) -> tensor<1x4xf32>
    "onnx.Return"(%tanh) : (tensor<1x4xf32>) -> ()
  }
}
