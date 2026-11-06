module {
  func.func @main_graph(%input: tensor<1x4xf32>) -> tensor<1x4xf32> {
    %fillShape = "onnx.Constant"() {
      value = dense<[1, 4]> : tensor<2xi64>
    } : () -> tensor<2xi64>
    %unusedFill = "onnx.ConstantOfShape"(%fillShape) {
      value = dense<0.0> : tensor<1xf32>
    } : (tensor<2xi64>) -> tensor<?x?xf32>
    %columns = "onnx.Constant"() {
      value = dense<[0, 1, 2, 3]> : tensor<4xi64>
    } : () -> tensor<4xi64>
    %rows = "onnx.Constant"() {
      value = dense<[[0], [1], [2], [3]]> : tensor<4x1xi64>
    } : () -> tensor<4x1xi64>
    %unusedGreater = "onnx.Greater"(%columns, %rows) :
        (tensor<4xi64>, tensor<4x1xi64>) -> tensor<4x4xi1>
    %trueValue = "onnx.Constant"() {
      value = dense<7> : tensor<i64>
    } : () -> tensor<i64>
    %falseValue = "onnx.Constant"() {
      value = dense<9> : tensor<i64>
    } : () -> tensor<i64>
    %unusedWhere = "onnx.Where"(%unusedGreater, %trueValue, %falseValue) :
        (tensor<4x4xi1>, tensor<i64>, tensor<i64>) -> tensor<4x4xi64>
    %triangularInput = "onnx.Constant"() {
      value = dense<1.0> : tensor<4x4xf32>
    } : () -> tensor<4x4xf32>
    %diagonal = "onnx.Constant"() {
      value = dense<1> : tensor<i64>
    } : () -> tensor<i64>
    %unusedTriangular = "onnx.Trilu"(%triangularInput, %diagonal) {
      upper = 1 : si64
    } : (tensor<4x4xf32>, tensor<i64>) -> tensor<4x4xf32>
    %sliceStart = "onnx.Constant"() {
      value = dense<1> : tensor<1xi64>
    } : () -> tensor<1xi64>
    %sliceEnd = "onnx.Constant"() {
      value = dense<3> : tensor<1xi64>
    } : () -> tensor<1xi64>
    %sliceAxis = "onnx.Constant"() {
      value = dense<0> : tensor<1xi64>
    } : () -> tensor<1xi64>
    %sliceStep = "onnx.Constant"() {
      value = dense<1> : tensor<1xi64>
    } : () -> tensor<1xi64>
    %unusedSlice = "onnx.Slice"(%columns, %sliceStart, %sliceEnd,
                                %sliceAxis, %sliceStep) :
        (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>,
         tensor<1xi64>) -> tensor<?xi64>
    %start = "onnx.Constant"() {
      value = dense<0> : tensor<1xi64>
    } : () -> tensor<1xi64>
    %limit = "onnx.Constant"() {
      value = dense<1> : tensor<1xi64>
    } : () -> tensor<1xi64>
    %delta = "onnx.Constant"() {
      value = dense<1> : tensor<1xi64>
    } : () -> tensor<1xi64>
    %axes = "onnx.Range"(%start, %limit, %delta) :
        (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
    %expanded = "onnx.Unsqueeze"(%input, %axes) :
        (tensor<1x4xf32>, tensor<?xi64>) -> tensor<1x1x4xf32>
    %squeezed = "onnx.Squeeze"(%expanded, %axes) :
        (tensor<1x1x4xf32>, tensor<?xi64>) -> tensor<1x4xf32>
    "onnx.Return"(%squeezed) : (tensor<1x4xf32>) -> ()
  }
}
