module {
  func.func @main_graph(%left: tensor<1x32x4x32xf32>,
                        %right: tensor<1x32x4x32xf32>)
      -> tensor<1x32x4x64xf32> {
    %result = "onnx.Concat"(%left, %right) {axis = 3 : si64} :
        (tensor<1x32x4x32xf32>, tensor<1x32x4x32xf32>) ->
        tensor<1x32x4x64xf32>
    "onnx.Return"(%result) : (tensor<1x32x4x64xf32>) -> ()
  }
}
