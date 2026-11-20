// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph --udoncl-materialize-onnx-host-globals --udoncl-materialize-onnx-host-model-registry --udoncl-attach-onnx-host-metadata --udoncl-finalize-onnx-host --udoncl-cleanup-onnx-host-source --canonicalize %s | %FileCheck %s
// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph --udoncl-materialize-onnx-host-globals --udoncl-materialize-onnx-host-model-registry --udoncl-attach-onnx-host-metadata --udoncl-finalize-onnx-host --udoncl-cleanup-onnx-host-source --canonicalize %s | %vrc-opt --vrc-assign-synthesized-kernel-ids=first-id=4096 --vrc-prove-synthesized-gather-kernels | %FileCheck %s --check-prefix=PROVEN
// RUN: not %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph=bank-sequence-state=false --udoncl-materialize-onnx-host-globals --udoncl-materialize-onnx-host-model-registry --udoncl-attach-onnx-host-metadata --udoncl-finalize-onnx-host %s 2>&1 | %FileCheck %s --check-prefix=COPIED

// A sequence state of capacity 128 with four heads of 64: the present holds
// 32768 words, beyond the 16384 of the library Concat and of the Slice of a
// copying advance. The Concat runs a synthesized kernel of its own in every
// bank, and banking needs no Slice. Without banks the state is rejected.

module attributes {udoncl.onnx_sequence_states = [
    {axis = 2 : i64, capacity = 128 : i64, input_index = 1 : i64,
     output_index = 0 : i64, step = 1 : i64}]} {
  func.func private @clEnqueueMarkerWithWaitList(
      !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32

  func.func private @__udon_onnx_concat(
      tensor<1x4x127x64xf32>, tensor<1x4x1x64xf32>) -> tensor<1x4x128x64xf32>
      attributes {
        udon.onnx.operation = "Concat",
        udon.onnx.provider = "udononnxConcat"
      }

  func.func @main_graph(
      %token: tensor<1x4x1x64xf32> {onnx.name = "token"},
      %past: tensor<1x4x127x64xf32> {onnx.name = "past"})
      -> (tensor<1x4x128x64xf32> {onnx.name = "present"}) {
    %present = func.call @__udon_onnx_concat(%past, %token)
        {udon.onnx.attributes = {axis = 2 : si64}}
        : (tensor<1x4x127x64xf32>, tensor<1x4x1x64xf32>)
        -> tensor<1x4x128x64xf32>
    return %present : tensor<1x4x128x64xf32>
  }
}

// CHECK-LABEL: func.func @udon_onnx_binding_enqueue_advance_sequence_state
// CHECK-NOT: call @__vrc_opencl_
// CHECK: return
// CHECK-LABEL: func.func @udononnxGeneratedEnqueueRun
// CHECK: call @__vrc_opencl_dispatch_udononnx_fused_0(%[[PRESENT:[^,]*]], %[[TOKEN:[^,]*]], %[[PAST:[^)]*]])
// CHECK: call @__vrc_opencl_dispatch_udononnx_fused_0(%[[PAST]], %[[TOKEN]], %[[PRESENT]])
// CHECK: call @__vrc_opencl_dispatch_udononnx_fused_1(%[[PAST]], %[[TOKEN]], %[[PRESENT]])
// CHECK-NOT: call @__vrc_opencl_dispatch_
// CHECK: gpu.func @udononnx_fused_0(%{{.*}}: memref<32768xf32>, %{{.*}}: memref<256xf32>, %{{.*}}: memref<32768xf32>)
// CHECK-SAME: known_grid_size = array<i32: 8192, 4, 1>
// CHECK: gpu.func @udononnx_fused_1(%{{.*}}: memref<32512xf32>, %{{.*}}: memref<256xf32>, %{{.*}}: memref<32768xf32>)

// PROVEN: gpu.func @udononnx_fused_0({{.*}}vrc.gather_proven
// PROVEN: gpu.func @udononnx_fused_1({{.*}}vrc.gather_proven

// COPIED: ONNX sequence state that is not banked exceeds the 16384 elements of its copying advance
