module {
  func.func @main_graph(
      %query: tensor<1x32x1x64xf32>,
      %keys: tensor<1x32x64x64xf32>) -> tensor<1x32x1x64xf32> {
    %result = "onnx.MatMul"(%query, %keys) :
        (tensor<1x32x1x64xf32>, tensor<1x32x64x64xf32>)
        -> tensor<1x32x1x64xf32>
    "onnx.Return"(%result) : (tensor<1x32x1x64xf32>) -> ()
  }
}
