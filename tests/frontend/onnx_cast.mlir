module {
  func.func @main_graph(%input: tensor<1x4xi64>) -> tensor<1x4xf32> {
    %cast = "onnx.Cast"(%input) {
      saturate = 1 : si64,
      to = f32
    } : (tensor<1x4xi64>) -> tensor<1x4xf32>
    %squared = "onnx.Mul"(%cast, %cast) :
        (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
    %shape = "onnx.Constant"() {
      value = dense<[1, 1, 4]> : tensor<3xi64>
    } : () -> tensor<3xi64>
    %expanded = "onnx.Expand"(%squared, %shape) :
        (tensor<1x4xf32>, tensor<3xi64>) -> tensor<1x1x4xf32>
    %axes = "onnx.Constant"() {
      value = dense<[0]> : tensor<1xi64>
    } : () -> tensor<1xi64>
    %result = "onnx.Squeeze"(%expanded, %axes) :
        (tensor<1x1x4xf32>, tensor<1xi64>) -> tensor<1x4xf32>
    %transposed = "onnx.Transpose"(%result) {
      perm = [1, 0]
    } : (tensor<1x4xf32>) -> tensor<4x1xf32>
    %restored = "onnx.Transpose"(%transposed) {
      perm = [1, 0]
    } : (tensor<4x1xf32>) -> tensor<1x4xf32>
    %cosine = "onnx.Cos"(%restored) :
        (tensor<1x4xf32>) -> tensor<1x4xf32>
    %sine = "onnx.Sin"(%cosine) :
        (tensor<1x4xf32>) -> tensor<1x4xf32>
    %negative = "onnx.Neg"(%sine) :
        (tensor<1x4xf32>) -> tensor<1x4xf32>
    %sigmoid = "onnx.Sigmoid"(%negative) :
        (tensor<1x4xf32>) -> tensor<1x4xf32>
    "onnx.Return"(%sigmoid) : (tensor<1x4xf32>) -> ()
  }
}
