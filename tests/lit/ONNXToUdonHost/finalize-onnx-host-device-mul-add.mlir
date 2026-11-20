// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph --udoncl-materialize-onnx-host-globals --udoncl-materialize-onnx-host-model-registry --udoncl-attach-onnx-host-metadata --udoncl-finalize-onnx-host --udoncl-cleanup-onnx-host-source %s | %FileCheck %s

// A Device Mul read only by one Add of the same shape is folded into the
// Add's OpTensor dispatch: operation 1 (multiply) with beta 1 adds the other
// Add operand as the previous value. A product with a second reader and an
// Add whose other operand is broadcast keep their own dispatches.

func.func private @clEnqueueMarkerWithWaitList(
    !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32

func.func private @__udon_onnx_mul(
    tensor<1x4x1x8xf32>, tensor<1x1x1x8xf32>) -> tensor<1x4x1x8xf32>
    attributes {
      udon.onnx.operation = "Mul",
      udon.onnx.provider = "udonatenMul"
    }

func.func private @__udon_onnx_add(
    tensor<1x4x1x8xf32>, tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
    attributes {
      udon.onnx.operation = "Add",
      udon.onnx.provider = "udonatenAdd"
    }

func.func private @__udon_onnx_add_row(
    tensor<1x4x1x8xf32>, tensor<1x1x1x8xf32>) -> tensor<1x4x1x8xf32>
    attributes {
      udon.onnx.operation = "Add",
      udon.onnx.provider = "udonatenAdd"
    }

func.func @main_graph(
    %x: tensor<1x4x1x8xf32> {onnx.name = "x"},
    %y: tensor<1x4x1x8xf32> {onnx.name = "y"},
    %cos: tensor<1x1x1x8xf32> {onnx.name = "cos"},
    %sin: tensor<1x1x1x8xf32> {onnx.name = "sin"})
    -> (tensor<1x4x1x8xf32> {onnx.name = "rotated"},
        tensor<1x4x1x8xf32> {onnx.name = "shared"},
        tensor<1x4x1x8xf32> {onnx.name = "row"}) {
  %x_cos = func.call @__udon_onnx_mul(%x, %cos)
      : (tensor<1x4x1x8xf32>, tensor<1x1x1x8xf32>) -> tensor<1x4x1x8xf32>
  %y_sin = func.call @__udon_onnx_mul(%y, %sin)
      : (tensor<1x4x1x8xf32>, tensor<1x1x1x8xf32>) -> tensor<1x4x1x8xf32>
  %rotated = func.call @__udon_onnx_add(%x_cos, %y_sin)
      : (tensor<1x4x1x8xf32>, tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %twice = func.call @__udon_onnx_mul(%y, %cos)
      : (tensor<1x4x1x8xf32>, tensor<1x1x1x8xf32>) -> tensor<1x4x1x8xf32>
  %shared = func.call @__udon_onnx_add(%twice, %twice)
      : (tensor<1x4x1x8xf32>, tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %x_sin = func.call @__udon_onnx_mul(%x, %sin)
      : (tensor<1x4x1x8xf32>, tensor<1x1x1x8xf32>) -> tensor<1x4x1x8xf32>
  %row = func.call @__udon_onnx_add_row(%x_sin, %cos)
      : (tensor<1x4x1x8xf32>, tensor<1x1x1x8xf32>) -> tensor<1x4x1x8xf32>
  return %rotated, %shared, %row
      : tensor<1x4x1x8xf32>, tensor<1x4x1x8xf32>, tensor<1x4x1x8xf32>
}

// CHECK-LABEL: func.func @udononnxGeneratedEnqueueRun

// x * cos is the previous value of the folded y * sin + (x * cos).
// CHECK: %[[MUL:.*]] = arith.constant 257 : i32
// CHECK-NEXT: arith.constant 1.000000e+00 : f32
// CHECK-NEXT: arith.constant 1.000000e+00 : f32
// CHECK-NEXT: %[[NO_PREVIOUS:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT: call @__vrc_opencl_dispatch_udondnn_large_op_tensor(%[[X:[^,]*]], %[[COS:[^,]*]], %[[X]], %[[X_COS:[^,]*]],
// CHECK-SAME: %[[MUL]], %{{[^,]*}}, %{{[^,]*}}, %[[NO_PREVIOUS]])
// CHECK: %[[FOLDED:.*]] = arith.constant 257 : i32
// CHECK-NEXT: arith.constant 1.000000e+00 : f32
// CHECK-NEXT: arith.constant 1.000000e+00 : f32
// CHECK-NEXT: %[[BETA:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-NEXT: call @__vrc_opencl_dispatch_udondnn_large_op_tensor(%[[Y:[^,]*]], %[[SIN:[^,]*]], %[[X_COS]], %[[ROTATED:[^,]*]],
// CHECK-SAME: %[[FOLDED]], %{{[^,]*}}, %{{[^,]*}}, %[[BETA]])

// A product read twice by its Add keeps its Mul dispatch.
// CHECK: call @__vrc_opencl_dispatch_udondnn_large_op_tensor(%[[Y]], %[[COS]], %[[Y]], %[[TWICE:[^,]*]],
// CHECK: %[[ADD:.*]] = arith.constant 256 : i32
// CHECK: call @__vrc_opencl_dispatch_udondnn_large_op_tensor(%[[TWICE]], %[[TWICE]], %[[TWICE]], %{{[^,]*}},
// CHECK-SAME: %[[ADD]],

// A broadcast Add operand cannot be the previous value.
// CHECK: call @__vrc_opencl_dispatch_udondnn_large_op_tensor(%[[X]], %[[SIN]], %[[X]], %[[X_SIN:[^,]*]],
// CHECK: %[[ROW_ADD:.*]] = arith.constant 256 : i32
// CHECK: call @__vrc_opencl_dispatch_udondnn_large_op_tensor(%[[X_SIN]], %[[COS]], %[[X_SIN]], %{{[^,]*}},
// CHECK-SAME: %[[ROW_ADD]],
// CHECK-NOT: call @__vrc_opencl_dispatch_udondnn_large_op_tensor
