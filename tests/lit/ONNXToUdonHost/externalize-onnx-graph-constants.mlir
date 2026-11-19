// RUN: %vrc-opt --udoncl-externalize-onnx-graph-constants %s | %FileCheck %s

// Graph-local dense constants (an onnx.Constant node without an initializer
// origin, folded integer indices, and a folded Cast result) become immutable
// static tensors whose ids continue after the initializer records. Their
// values stay inline so Host finalization can still fold them. Scalars and
// statically empty tensors are not tensor payloads and stay untouched.

// CHECK: module attributes {udoncl.static_tensors = [{id = 0 : i32}
// CHECK-SAME: id = 1 : i32
// CHECK-SAME: name = "constant_1"
// CHECK-SAME: data_type = "int64", id = 2 : i32
// CHECK-SAME: name = "constant_2"
// CHECK-SAME: id = 3 : i32
// CHECK-SAME: name = "constant_3"
// CHECK-NOT: id = 4 : i32
// CHECK-DAG: func.func private @__udoncl_static_tensor_1() -> tensor<1x1x2x2xf32> attributes {udoncl.static_tensor = {{{.*}}data = dense<{{.*}}> : tensor<1x1x2x2xf32>
// CHECK-DAG: func.func private @__udoncl_static_tensor_2() -> tensor<1x2x1xi64>
// CHECK-DAG: func.func private @__udoncl_static_tensor_3() -> tensor<2xf32>
// CHECK-LABEL: func.func @main_graph
// CHECK: %[[MASK:.*]] = call @__udoncl_static_tensor_1()
// CHECK: %[[INDICES:.*]] = call @__udoncl_static_tensor_2()
// CHECK: %[[CAST:.*]] = call @__udoncl_static_tensor_3()
// CHECK: arith.constant 7 : i64
// CHECK: arith.constant dense<> : tensor<0xi64>
// CHECK-NOT: onnx.Constant
module attributes {udoncl.static_tensors = [{id = 0 : i32}]} {
  func.func @main_graph(%x: tensor<1x1x2x2xf32>)
      -> (tensor<1x1x2x2xf32>, tensor<1x2x1xi64>, tensor<2xf32>, i64, tensor<0xi64>) {
    %mask = onnx.Constant dense<[[[[0.0, -1.0], [0.0, 0.0]]]]> : tensor<1x1x2x2xf32>
    %indices = arith.constant dense<[[[0], [1]]]> : tensor<1x2x1xi64>
    %cast = arith.constant dense<[1.0, 0.0]> : tensor<2xf32>
    %scalar = arith.constant 7 : i64
    %empty = arith.constant dense<> : tensor<0xi64>
    %y = "onnx.Add"(%x, %mask) : (tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32>
    return %y, %indices, %cast, %scalar, %empty
        : tensor<1x1x2x2xf32>, tensor<1x2x1xi64>, tensor<2xf32>, i64, tensor<0xi64>
  }
}
