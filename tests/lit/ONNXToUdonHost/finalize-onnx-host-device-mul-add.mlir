// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph --udoncl-materialize-onnx-host-globals --udoncl-materialize-onnx-host-model-registry --udoncl-attach-onnx-host-metadata --udoncl-finalize-onnx-host --udoncl-cleanup-onnx-host-source %s | %FileCheck %s
// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph=fuse-device-nodes=false --udoncl-materialize-onnx-host-globals --udoncl-materialize-onnx-host-model-registry --udoncl-attach-onnx-host-metadata --udoncl-finalize-onnx-host --udoncl-cleanup-onnx-host-source %s | %FileCheck %s --check-prefix=UNFUSED

// A Device Mul whose product only an Add reads is an interior node of the
// Add's fusion group like any other fusible producer: the Add's synthesized
// kernel computes the product and the sum, and the product has no buffer.
// There is no separate Mul+Add path through the OpTensor kernel.

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
// CHECK-NOT: call @__vrc_opencl_dispatch_udondnn_large_op_tensor

// x * cos + y * sin
// CHECK: call @__vrc_opencl_dispatch_udononnx_fused_0(%[[X:[^,]*]], %[[COS:[^,]*]], %[[Y:[^,]*]], %[[SIN:[^,]*]], %{{[^,]*}}) : (i32, i32, i32, i32, i32) -> i32
// A product that its Add reads twice is evaluated once.
// CHECK: call @__vrc_opencl_dispatch_udononnx_fused_1(%[[Y]], %[[COS]], %{{[^,]*}}) : (i32, i32, i32) -> i32
// The other Add operand may be broadcast.
// CHECK: call @__vrc_opencl_dispatch_udononnx_fused_2(%[[X]], %[[SIN]], %[[COS]], %{{[^,]*}}) : (i32, i32, i32, i32) -> i32
// CHECK-NOT: call @__vrc_opencl_dispatch_

// CHECK-LABEL: gpu.module @udononnx_fused_kernels {
// The products are rounded before they are added: two mulf and one addf
// without fast-math flags, never a fused multiply-add.
// CHECK: gpu.func @udononnx_fused_0(%[[X:.*]]: memref<32xf32>, %[[COS:.*]]: memref<8xf32>, %[[Y:.*]]: memref<32xf32>, %[[SIN:.*]]: memref<8xf32>, %{{.*}}: memref<8192xf32>) kernel
// CHECK: %[[X_VALUE:.*]] = memref.load %[[X]][
// CHECK: %[[COS_VALUE:.*]] = memref.load %[[COS]][
// CHECK: %[[X_COS:.*]] = arith.mulf %[[X_VALUE]], %[[COS_VALUE]] : f32
// CHECK: %[[Y_VALUE:.*]] = memref.load %[[Y]][
// CHECK: %[[SIN_VALUE:.*]] = memref.load %[[SIN]][
// CHECK: %[[Y_SIN:.*]] = arith.mulf %[[Y_VALUE]], %[[SIN_VALUE]] : f32
// CHECK: %[[SUM:.*]] = arith.addf %[[X_COS]], %[[Y_SIN]] : f32
// CHECK: scf.yield %[[SUM]] : f32

// CHECK: gpu.func @udononnx_fused_1(%[[Y:.*]]: memref<32xf32>, %[[COS:.*]]: memref<8xf32>, %{{.*}}: memref<8192xf32>) kernel
// CHECK-SAME: vrc.fragment_cost = {loads = 2 : i64}
// CHECK: %[[TWICE:.*]] = arith.mulf
// CHECK-NOT: arith.mulf
// CHECK: arith.addf %[[TWICE]], %[[TWICE]] : f32

// CHECK: gpu.func @udononnx_fused_2(%[[X:.*]]: memref<32xf32>, %[[SIN:.*]]: memref<8xf32>, %[[COS:.*]]: memref<8xf32>, %{{.*}}: memref<8192xf32>) kernel
// CHECK: %[[X_SIN:.*]] = arith.mulf
// CHECK: %[[COS_VALUE:.*]] = memref.load %[[COS]][
// CHECK: arith.addf %[[X_SIN]], %[[COS_VALUE]] : f32

// Unfused, every Mul and Add is its own OpTensor dispatch with beta 0.
// UNFUSED-LABEL: func.func @udononnxGeneratedEnqueueRun
// UNFUSED-COUNT-7: call @__vrc_opencl_dispatch_udondnn_large_op_tensor(
// UNFUSED-NOT: call @__vrc_opencl_dispatch_
// UNFUSED-NOT: gpu.module
