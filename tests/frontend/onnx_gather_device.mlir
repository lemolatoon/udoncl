#initializer = loc("Initializer_embedding.weight")
module {
  func.func @main_graph(%indices: tensor<1x3xi64>) -> tensor<1x3x4xf32> {
    %data = "onnx.Constant"() {
      value = dense<[
        [0.0, 1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0, 7.0],
        [8.0, 9.0, 10.0, 11.0],
        [12.0, 13.0, 14.0, 15.0],
        [16.0, 17.0, 18.0, 19.0]
      ]> : tensor<5x4xf32>
    } : () -> tensor<5x4xf32> loc(#initializer)
    %result = "onnx.Gather"(%data, %indices) {axis = 0 : si64} :
        (tensor<5x4xf32>, tensor<1x3xi64>) -> tensor<1x3x4xf32>
    "onnx.Return"(%result) : (tensor<1x3x4xf32>) -> ()
  }
}
