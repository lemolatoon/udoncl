module {
  func.func @main_graph(%input: tensor<1x32x4x64xf32>)
      -> tensor<1x32x4x32xf32> {
    %starts = "onnx.Constant"() {
      value = dense<[32]> : tensor<1xi64>
    } : () -> tensor<1xi64>
    %ends = "onnx.Constant"() {
      value = dense<[64]> : tensor<1xi64>
    } : () -> tensor<1xi64>
    %axes = "onnx.Constant"() {
      value = dense<[3]> : tensor<1xi64>
    } : () -> tensor<1xi64>
    %steps = "onnx.Constant"() {
      value = dense<[1]> : tensor<1xi64>
    } : () -> tensor<1xi64>
    %result = "onnx.Slice"(%input, %starts, %ends, %axes, %steps) :
        (tensor<1x32x4x64xf32>, tensor<1xi64>, tensor<1xi64>,
         tensor<1xi64>, tensor<1xi64>) -> tensor<1x32x4x32xf32>
    "onnx.Return"(%result) : (tensor<1x32x4x32xf32>) -> ()
  }
}
