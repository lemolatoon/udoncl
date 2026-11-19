// RUN: %vrc-opt --udoncl-infer-onnx-shapes %s | %FileCheck %s

// A Cast of a statically known integer tensor to floating point is folded.
// Booleans convert to 0/1, signed integers keep their value, and a Cast whose
// input is not statically known remains.

module {
  func.func @main_graph(%dynamic: tensor<2xi64>)
      -> (tensor<2x2xf32>, tensor<3xf32>, tensor<2xf32>) {
    %rows = "onnx.Constant"() {value = dense<[[0], [1]]> : tensor<2x1xi64>}
        : () -> tensor<2x1xi64>
    %columns = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64>}
        : () -> tensor<2xi64>
    %mask = "onnx.Greater"(%columns, %rows)
        : (tensor<2xi64>, tensor<2x1xi64>) -> tensor<2x2xi1>
    %mask_f32 = "onnx.Cast"(%mask) {saturate = 1 : si64, to = f32}
        : (tensor<2x2xi1>) -> tensor<2x2xf32>
    %signed = "onnx.Constant"() {value = dense<[-3, 0, 7]> : tensor<3xi64>}
        : () -> tensor<3xi64>
    %signed_f32 = "onnx.Cast"(%signed) {saturate = 1 : si64, to = f32}
        : (tensor<3xi64>) -> tensor<3xf32>
    %dynamic_f32 = "onnx.Cast"(%dynamic) {saturate = 1 : si64, to = f32}
        : (tensor<2xi64>) -> tensor<2xf32>
    "onnx.Return"(%mask_f32, %signed_f32, %dynamic_f32)
        : (tensor<2x2xf32>, tensor<3xf32>, tensor<2xf32>) -> ()
  }
}

// CHECK-LABEL: func.func @main_graph
// CHECK-DAG: %[[MASK:.*]] = arith.constant dense<{{\[\[}}0.000000e+00, 1.000000e+00], [0.000000e+00, 0.000000e+00]]> : tensor<2x2xf32>
// CHECK-DAG: %[[SIGNED:.*]] = arith.constant dense<[-3.000000e+00, 0.000000e+00, 7.000000e+00]> : tensor<3xf32>
// CHECK: %[[DYNAMIC:.*]] = "onnx.Cast"(%arg0)
// CHECK: onnx.Return %[[MASK]], %[[SIGNED]], %[[DYNAMIC]]
