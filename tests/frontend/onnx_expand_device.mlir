module {
  func.func @main_graph(%input: tensor<1x4x1x4x64xf32>)
      -> tensor<1x4x8x4x64xf32> {
    %shape = "onnx.Constant"() {
      value = dense<[1, 4, 8, 4, 64]> : tensor<5xi64>
    } : () -> tensor<5xi64>
    %result = "onnx.Expand"(%input, %shape) :
        (tensor<1x4x1x4x64xf32>, tensor<5xi64>) ->
        tensor<1x4x8x4x64xf32>
    "onnx.Return"(%result) : (tensor<1x4x8x4x64xf32>) -> ()
  }
}
