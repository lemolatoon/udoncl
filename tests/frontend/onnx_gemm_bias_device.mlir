module {
  func.func @main_graph(%input: tensor<1x144xf32>,
                        %weight: tensor<144x10xf32>,
                        %bias: tensor<10xf32>) -> tensor<1x10xf32> {
    %output = "onnx.Gemm"(%input, %weight, %bias)
        {alpha = 1.0 : f32, beta = 1.0 : f32,
         transA = 0 : si64, transB = 0 : si64} :
        (tensor<1x144xf32>, tensor<144x10xf32>, tensor<10xf32>) ->
        tensor<1x10xf32>
    "onnx.Return"(%output) : (tensor<1x10xf32>) -> ()
  }
}
