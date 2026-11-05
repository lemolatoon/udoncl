module {
  func.func @main_graph(%left: tensor<1x4xf32>, %right: tensor<1x4xf32>) -> tensor<1x4xf32> {
    %sum = "onnx.Add"(%left, %right) : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
    "onnx.Return"(%sum) : (tensor<1x4xf32>) -> ()
  }
  func.func @matmul_nbits(%a: tensor<1x4x8xf32>, %b: tensor<2x1x16xui8>,
                        %scales: tensor<2xf32>) {
  %0 = "onnx.Custom"(%a, %b, %scales) <{function_name = "MatMulNBits"}>
      {K = 8 : si64, N = 16 : si64, bits = 4 : si64,
       block_size = 32 : si64, domain_name = "com.microsoft"} :
      (tensor<1x4x8xf32>, tensor<2x1x16xui8>, tensor<2xf32>) -> tensor<*xf32>
    "onnx.Return"() : () -> ()
  }
  func.func @rms_norm(%input: tensor<1x4x8xf32>, %weight: tensor<8xf32>,
                      %epsilon: tensor<f32>) -> tensor<*xf32> {
    %0 = "onnx.Custom"(%input, %weight, %epsilon)
        <{function_name = "RmsNorm"}>
        {domain_name = "com.lemolatoon.udoncl"} :
        (tensor<1x4x8xf32>, tensor<8xf32>, tensor<f32>) -> tensor<*xf32>
    "onnx.Return"(%0) : (tensor<*xf32>) -> ()
  }
}
