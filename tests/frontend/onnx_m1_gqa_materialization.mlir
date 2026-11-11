module {
  func.func @main_graph(%input: tensor<1x4x1x64x64xf32>,
                        %scale: tensor<1xf32>)
      -> tensor<1x32x64x64xf32> {
    %expanded_shape = "onnx.Constant"() {
      value = dense<[1, 4, 8, 64, 64]> : tensor<5xi64>
    } : () -> tensor<5xi64>
    %expanded = "onnx.Expand"(%input, %expanded_shape) :
        (tensor<1x4x1x64x64xf32>, tensor<5xi64>) ->
        tensor<1x4x8x64x64xf32>
    %reshaped_shape = "onnx.Constant"() {
      value = dense<[1, 32, 64, 64]> : tensor<4xi64>
    } : () -> tensor<4xi64>
    %reshaped = "onnx.Reshape"(%expanded, %reshaped_shape) {
      allowzero = 0 : si64
    } : (tensor<1x4x8x64x64xf32>, tensor<4xi64>) ->
        tensor<1x32x64x64xf32>
    %transposed = "onnx.Transpose"(%reshaped) {
      perm = [0, 1, 3, 2]
    } : (tensor<1x32x64x64xf32>) -> tensor<1x32x64x64xf32>
    %result = "onnx.Mul"(%transposed, %scale) :
        (tensor<1x32x64x64xf32>, tensor<1xf32>) ->
        tensor<1x32x64x64xf32>
    "onnx.Return"(%result) : (tensor<1x32x64x64xf32>) -> ()
  }
}
