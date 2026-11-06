module {
  func.func @main_graph(%input: tensor<1x4x32x64xf32>)
      -> tensor<1x32x4x64xf32> {
    %result = "onnx.Transpose"(%input) {
      perm = [0, 2, 1, 3]
    } : (tensor<1x4x32x64xf32>) -> tensor<1x32x4x64xf32>
    "onnx.Return"(%result) : (tensor<1x32x4x64xf32>) -> ()
  }
}
