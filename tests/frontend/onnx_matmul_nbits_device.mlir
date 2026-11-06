#packed = loc("Initializer_packed")
#scales = loc("Initializer_scales")
module {
  func.func @main_graph(%activation: tensor<1x32xf32> {onnx.name = "activation"})
      -> tensor<1x32xf32> {
    %packed = "onnx.Constant"() {
      value = dense<136> : tensor<32x1x16xui8>
    } : () -> tensor<32x1x16xui8> loc(#packed)
    %scales = "onnx.Constant"() {
      value = dense<0.5> : tensor<32xf32>
    } : () -> tensor<32xf32> loc(#scales)
    %output = "onnx.Custom"(%activation, %packed, %scales)
        <{function_name = "MatMulNBits"}>
        {K = 32 : si64, N = 32 : si64, bits = 4 : si64,
         block_size = 32 : si64, domain_name = "com.microsoft"} :
        (tensor<1x32xf32>, tensor<32x1x16xui8>, tensor<32xf32>)
        -> tensor<1x32xf32>
    "onnx.Return"(%output) : (tensor<1x32xf32>) -> ()
  }
}
