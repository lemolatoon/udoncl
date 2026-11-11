module {
  func.func @main_graph(
      %token: tensor<1x4x1x64xf32>,
      %past_key: tensor<1x4x7x64xf32>,
      %past_value: tensor<1x4x7x64xf32>,
      %not_state: tensor<1x2xf32>)
      -> (tensor<1x4x8x64xf32>, tensor<1x2xf32>,
          tensor<1x4x8x64xf32>) {
    %present_key = "onnx.Concat"(%past_key, %token) {axis = 2 : si64} :
        (tensor<1x4x7x64xf32>, tensor<1x4x1x64xf32>)
        -> tensor<1x4x8x64xf32>
    %present_value = "onnx.Concat"(%past_value, %token) {axis = 2 : si64} :
        (tensor<1x4x7x64xf32>, tensor<1x4x1x64xf32>)
        -> tensor<1x4x8x64xf32>
    "onnx.Return"(%present_key, %not_state, %present_value) :
        (tensor<1x4x8x64xf32>, tensor<1x2xf32>,
         tensor<1x4x8x64xf32>) -> ()
  }
}
