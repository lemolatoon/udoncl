// RUN: %vrc-opt --udoncl-canonicalize-onnx-library-patterns %s | %FileCheck %s

// The RMS normalization subgraph becomes one RmsNorm library operation, and
// the replaced generic operations are erased instead of being left for later
// lowering as dead Host or Device work.

module {
  func.func @main_graph(%input: tensor<1x4x8xf32>) -> tensor<1x4x8xf32> {
    %two = "onnx.Constant"() {
      value = dense<2.0> : tensor<f32>
    } : () -> tensor<f32>
    %epsilon = "onnx.Constant"() {
      value = dense<1.0e-5> : tensor<f32>
    } : () -> tensor<f32>
    %one = "onnx.Constant"() {
      value = dense<1.0> : tensor<f32>
    } : () -> tensor<f32>
    %weight = "onnx.Constant"() {
      value = dense<1.0> : tensor<8xf32>
    } : () -> tensor<8xf32>
    %square = "onnx.Pow"(%input, %two) :
        (tensor<1x4x8xf32>, tensor<f32>) -> tensor<1x4x8xf32>
    %mean = "onnx.ReduceMeanV13"(%square) {
      axes = [-1],
      keepdims = 1 : si64
    } : (tensor<1x4x8xf32>) -> tensor<1x4x1xf32>
    %biased = "onnx.Add"(%mean, %epsilon) :
        (tensor<1x4x1xf32>, tensor<f32>) -> tensor<1x4x1xf32>
    %root = "onnx.Sqrt"(%biased) :
        (tensor<1x4x1xf32>) -> tensor<1x4x1xf32>
    %reciprocal = "onnx.Div"(%one, %root) :
        (tensor<f32>, tensor<1x4x1xf32>) -> tensor<1x4x1xf32>
    %normalized = "onnx.Mul"(%input, %reciprocal) :
        (tensor<1x4x8xf32>, tensor<1x4x1xf32>) -> tensor<1x4x8xf32>
    %result = "onnx.Mul"(%normalized, %weight) :
        (tensor<1x4x8xf32>, tensor<8xf32>) -> tensor<1x4x8xf32>
    "onnx.Return"(%result) : (tensor<1x4x8xf32>) -> ()
  }
}

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: onnx.Pow
// CHECK-NOT: onnx.ReduceMeanV13
// CHECK-NOT: onnx.Sqrt
// CHECK-NOT: onnx.Div
// CHECK: %[[NORM:.*]] = "onnx.Custom"(%arg0, %{{.*}}, %{{.*}}) <{function_name = "RmsNorm"}> {domain_name = "com.lemolatoon.udoncl"}
// CHECK-NOT: onnx.Mul
// CHECK: onnx.Return %[[NORM]]
