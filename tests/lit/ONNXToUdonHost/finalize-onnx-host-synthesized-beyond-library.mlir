// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph %s | %FileCheck %s --check-prefix=PLAN
// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph --udoncl-materialize-onnx-host-globals --udoncl-materialize-onnx-host-model-registry --udoncl-attach-onnx-host-metadata --udoncl-finalize-onnx-host --udoncl-cleanup-onnx-host-source --canonicalize %s | %FileCheck %s
// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph --udoncl-materialize-onnx-host-globals --udoncl-materialize-onnx-host-model-registry --udoncl-attach-onnx-host-metadata --udoncl-finalize-onnx-host --udoncl-cleanup-onnx-host-source --canonicalize %s | %vrc-opt --vrc-assign-synthesized-kernel-ids=first-id=4096 --vrc-prove-synthesized-gather-kernels | %FileCheck %s --check-prefix=PROVEN
// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph=fuse-device-nodes=false --udoncl-materialize-onnx-host-globals --udoncl-materialize-onnx-host-model-registry --udoncl-attach-onnx-host-metadata --udoncl-finalize-onnx-host --udoncl-cleanup-onnx-host-source --canonicalize %s | %FileCheck %s --check-prefix=UNFUSED

// Per-element, re-indexing and Softmax nodes beyond the element limits of
// their library kernels still run on the Device, as synthesized kernels that
// launch the rows their result needs. A sequence state of capacity 4096 with
// four heads of 64 holds 1048576 words per present (the library Concat
// addresses 16384, the library Transpose and Mul 131072): its banked Concat
// launches 8192 x 128 fragments. The transposed, scaled present is one group
// of the same size, and a Softmax over rows of 4096 (beyond the 512 of the
// library kernel) runs in two passes with its mask Add fused. Without fusion
// every such node is a synthesized kernel of its own.

module attributes {udoncl.onnx_sequence_states = [
    {axis = 2 : i64, capacity = 4096 : i64, input_index = 1 : i64,
     output_index = 0 : i64, step = 1 : i64}]} {
  func.func private @clEnqueueMarkerWithWaitList(
      !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32

  func.func private @__udon_onnx_concat(
      tensor<1x4x4095x64xf32>, tensor<1x4x1x64xf32>) -> tensor<1x4x4096x64xf32>
      attributes {
        udon.onnx.operation = "Concat",
        udon.onnx.provider = "udononnxConcat"
      }
  func.func private @__udon_onnx_transpose(
      tensor<1x4x4096x64xf32>) -> tensor<1x4x64x4096xf32>
      attributes {
        udon.onnx.operation = "Transpose",
        udon.onnx.provider = "udononnxTranspose"
      }
  func.func private @__udon_onnx_mul(
      tensor<1x4x64x4096xf32>, tensor<f32>) -> tensor<1x4x64x4096xf32>
      attributes {
        udon.onnx.operation = "Mul",
        udon.onnx.provider = "udonatenMul"
      }
  func.func private @__udon_onnx_add(
      tensor<1x32x1x4096xf32>, tensor<1x1x1x4096xf32>)
      -> tensor<1x32x1x4096xf32>
      attributes {
        udon.onnx.operation = "Add",
        udon.onnx.provider = "udonatenAdd"
      }
  func.func private @__udon_onnx_softmax(
      tensor<1x32x1x4096xf32>) -> tensor<1x32x1x4096xf32>
      attributes {
        udon.onnx.operation = "Softmax",
        udon.onnx.provider = "udondnnSoftmaxForward"
      }
  func.func private @__static_scale() -> tensor<f32>
      attributes {
        udoncl.static_tensor = {
          byte_length = 4 : i64, id = 0 : i64,
          data = dense<1.250000e-01> : tensor<f32>
        }
      }

  func.func @main_graph(
      %token: tensor<1x4x1x64xf32> {onnx.name = "token"},
      %past: tensor<1x4x4095x64xf32> {onnx.name = "past"},
      %scores: tensor<1x32x1x4096xf32> {onnx.name = "scores"},
      %mask: tensor<1x1x1x4096xf32> {onnx.name = "mask"})
      -> (tensor<1x4x4096x64xf32> {onnx.name = "present"},
          tensor<1x4x64x4096xf32> {onnx.name = "keys"},
          tensor<1x32x1x4096xf32> {onnx.name = "probabilities"}) {
    %scale = func.call @__static_scale() : () -> tensor<f32>
    %present = func.call @__udon_onnx_concat(%past, %token)
        {udon.onnx.attributes = {axis = 2 : si64}}
        : (tensor<1x4x4095x64xf32>, tensor<1x4x1x64xf32>)
        -> tensor<1x4x4096x64xf32>
    %transposed = func.call @__udon_onnx_transpose(%present)
        {udon.onnx.attributes = {perm = [0, 1, 3, 2]}}
        : (tensor<1x4x4096x64xf32>) -> tensor<1x4x64x4096xf32>
    %keys = func.call @__udon_onnx_mul(%transposed, %scale)
        : (tensor<1x4x64x4096xf32>, tensor<f32>) -> tensor<1x4x64x4096xf32>
    %masked = func.call @__udon_onnx_add(%scores, %mask)
        : (tensor<1x32x1x4096xf32>, tensor<1x1x1x4096xf32>)
        -> tensor<1x32x1x4096xf32>
    %probabilities = func.call @__udon_onnx_softmax(%masked)
        {udon.onnx.attributes = {axis = -1 : si64}}
        : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x4096xf32>
    return %present, %keys, %probabilities
        : tensor<1x4x4096x64xf32>, tensor<1x4x64x4096xf32>,
          tensor<1x32x1x4096xf32>
  }
}

// The whole graph runs on the Device: the Transpose joins the group of the
// scaled keys and the mask Add the group of the Softmax.
// PLAN: device_fusion_roots = array<i64: 0, 2, 2, 4, 4>
// PLAN-SAME: device_graph = true

// The Concat and the scaled keys read the present of the active bank; the
// statistics and elementwise passes of the Softmax follow.
// CHECK-LABEL: func.func @udononnxGeneratedEnqueueRun
// CHECK-NOT: call @__vrc_opencl_dispatch_udononnx_transpose
// CHECK-NOT: call @__vrc_opencl_dispatch_udonaten
// CHECK-NOT: call @__vrc_opencl_dispatch_udondnn
// CHECK: call @__vrc_opencl_dispatch_udononnx_fused_3(%[[SCORES:[^,]*]], %[[MASK:[^,]*]], %[[SCRATCH:[^)]*]])
// CHECK: call @__vrc_opencl_dispatch_udononnx_fused_2(
// CHECK: call @__vrc_opencl_dispatch_udononnx_fused_4(%[[SCORES]], %[[MASK]], %[[SCRATCH]], %{{[^)]*}})
// CHECK-LABEL: gpu.module @udononnx_fused_kernels
// The Concat and the scaled keys also take the zero rows of the state
// (finalize-onnx-host-sequence-zero-rows.mlir).
// CHECK: gpu.func @udononnx_fused_0(%{{.*}}: memref<1048576xf32>, %{{.*}}: memref<256xf32>, %{{.*}}: memref<1048576xf32>, %{{.*}}: i32)
// CHECK-SAME: known_grid_size = array<i32: 8192, 128, 1>
// CHECK-SAME: vrc.words = 1048576
// CHECK: gpu.func @udononnx_fused_2(%{{.*}}: memref<1048576xf32>, %{{.*}}: memref<1048576xf32>, %{{.*}}: i32)
// CHECK-SAME: known_grid_size = array<i32: 8192, 128, 1>
// CHECK: gpu.func @udononnx_fused_4(%{{.*}}: memref<131072xf32>, %{{.*}}: memref<4096xf32>, %{{.*}}: memref<8192xf32>, %{{.*}}: memref<131072xf32>)
// CHECK-SAME: known_grid_size = array<i32: 8192, 16, 1>

// PROVEN-COUNT-5: vrc.gather_proven

// UNFUSED: device_graph = true
// UNFUSED-LABEL: func.func @udononnxGeneratedEnqueueRun
// UNFUSED-NOT: call @__vrc_opencl_dispatch_udononnx_transpose
// UNFUSED-NOT: call @__vrc_opencl_dispatch_udonaten
// UNFUSED-NOT: call @__vrc_opencl_dispatch_udondnn
// UNFUSED: return
