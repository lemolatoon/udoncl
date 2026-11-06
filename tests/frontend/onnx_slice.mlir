module {
  func.func @main_graph(%input: tensor<2x4xf32>) -> tensor<2x2xf32> {
    %starts = "onnx.Constant"() {
      value = dense<1> : tensor<1xi64>
    } : () -> tensor<1xi64>
    %ends = "onnx.Constant"() {
      value = dense<4> : tensor<1xi64>
    } : () -> tensor<1xi64>
    %axes = "onnx.Constant"() {
      value = dense<1> : tensor<1xi64>
    } : () -> tensor<1xi64>
    %steps = "onnx.Constant"() {
      value = dense<2> : tensor<1xi64>
    } : () -> tensor<1xi64>
    %result = "onnx.Slice"(%input, %starts, %ends, %axes, %steps) :
        (tensor<2x4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>,
         tensor<1xi64>) -> tensor<2x2xf32>
    "onnx.Return"(%result) : (tensor<2x2xf32>) -> ()
  }
}
