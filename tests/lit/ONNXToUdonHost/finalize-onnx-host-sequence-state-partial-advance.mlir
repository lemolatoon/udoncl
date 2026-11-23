// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph %s | %FileCheck %s --check-prefix=PLAN
// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph --udoncl-materialize-onnx-host-globals --udoncl-materialize-onnx-host-model-registry --udoncl-attach-onnx-host-metadata --udoncl-finalize-onnx-host --udoncl-cleanup-onnx-host-source --canonicalize %s | %FileCheck %s
// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph --udoncl-materialize-onnx-host-globals --udoncl-materialize-onnx-host-model-registry --udoncl-attach-onnx-host-metadata --udoncl-finalize-onnx-host --udoncl-cleanup-onnx-host-source --canonicalize %s | %vrc-opt --vrc-assign-synthesized-kernel-ids=first-id=4096 --vrc-prove-synthesized-gather-kernels | %FileCheck %s --check-prefix=PROVEN
// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph=bank-sequence-state=false --udoncl-materialize-onnx-host-globals --udoncl-materialize-onnx-host-model-registry --udoncl-attach-onnx-host-metadata --udoncl-finalize-onnx-host --udoncl-cleanup-onnx-host-source --canonicalize %s | %FileCheck %s --check-prefix=COPIED

// The graph of finalize-onnx-host-sequence-state-banks.mlir specialized to
// a sequence of two rows: every run appends two rows to both states of
// capacity 4. A caller that ran only one real position (the other computed
// from padding) commits just that row with
// udon_onnx_binding_enqueue_advance_sequence_state_rows(1), and the next run
// reads its carried rows one row, not two, into the previous present; the
// padded row is never carried (sequenceStatePartialAdvanceStep in
// ONNXHostSequenceExtents.h).
module attributes {udoncl.onnx_sequence_states = [
    {axis = 2 : i64, capacity = 4 : i64, input_index = 1 : i64,
     output_index = 0 : i64, step = 2 : i64},
    {axis = 2 : i64, capacity = 4 : i64, input_index = 2 : i64,
     output_index = 1 : i64, step = 2 : i64}]} {
  func.func private @clEnqueueMarkerWithWaitList(
      !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32

  func.func private @__udon_onnx_neg(tensor<1x2x2x8xf32>)
      -> tensor<1x2x2x8xf32>
      attributes {
        udon.onnx.operation = "Neg",
        udon.onnx.provider = "udononnxNeg"
      }
  func.func private @__udon_onnx_concat(
      tensor<1x2x2x8xf32>, tensor<1x2x2x8xf32>) -> tensor<1x2x4x8xf32>
      attributes {
        udon.onnx.operation = "Concat",
        udon.onnx.provider = "udononnxConcat"
      }
  func.func private @__udon_onnx_mul_scalar(
      tensor<1x2x4x8xf32>, tensor<f32>) -> tensor<1x2x4x8xf32>
      attributes {
        udon.onnx.operation = "Mul",
        udon.onnx.provider = "udonatenMul"
      }
  func.func private @__udon_onnx_matmul(
      tensor<1x2x2x4xf32>, tensor<1x2x4x8xf32>) -> tensor<1x2x2x8xf32>
      attributes {
        udon.onnx.operation = "MatMul",
        udon.onnx.provider = "udonblasSgemm"
      }
  func.func private @__static_scale() -> tensor<f32>
      attributes {
        udoncl.static_tensor = {
          byte_length = 4 : i64, id = 0 : i64,
          data = dense<5.000000e-01> : tensor<f32>
        }
      }

  func.func @main_graph(
      %token: tensor<1x2x2x8xf32> {onnx.name = "token"},
      %past_key: tensor<1x2x2x8xf32> {onnx.name = "past_key"},
      %past_value: tensor<1x2x2x8xf32> {onnx.name = "past_value"},
      %weights: tensor<1x2x2x4xf32> {onnx.name = "weights"})
      -> (tensor<1x2x4x8xf32> {onnx.name = "present_key"},
          tensor<1x2x4x8xf32> {onnx.name = "present_value"},
          tensor<1x2x4x8xf32> {onnx.name = "scaled_keys"},
          tensor<1x2x2x8xf32> {onnx.name = "mixed"}) {
    %scale = func.call @__static_scale() : () -> tensor<f32>
    %negated = func.call @__udon_onnx_neg(%token)
        : (tensor<1x2x2x8xf32>) -> tensor<1x2x2x8xf32>
    %present_key = func.call @__udon_onnx_concat(%past_key, %negated)
        {udon.onnx.attributes = {axis = 2 : si64}}
        : (tensor<1x2x2x8xf32>, tensor<1x2x2x8xf32>) -> tensor<1x2x4x8xf32>
    %present_value = func.call @__udon_onnx_concat(%past_value, %token)
        {udon.onnx.attributes = {axis = -2 : si64}}
        : (tensor<1x2x2x8xf32>, tensor<1x2x2x8xf32>) -> tensor<1x2x4x8xf32>
    %scaled = func.call @__udon_onnx_mul_scalar(%present_key, %scale)
        : (tensor<1x2x4x8xf32>, tensor<f32>) -> tensor<1x2x4x8xf32>
    %mixed = func.call @__udon_onnx_matmul(%weights, %present_value)
        : (tensor<1x2x2x4xf32>, tensor<1x2x4x8xf32>) -> tensor<1x2x2x8xf32>
    return %present_key, %present_value, %scaled, %mixed
        : tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32>,
          tensor<1x2x2x8xf32>
  }
}

// Both states are banked, and the binding keeps the rows the last advance
// committed.
// PLAN: sequence_state_banks = array<i64: 0, 1>
// CHECK: sym_name = "__udon_onnx_sequence_state_advanced_rows"

// The plain advance commits a whole step: the next run reads two rows into
// the previous present, and the fill age, which counts committed rows, grows
// by two up to its limit carried + 1 = 3.
// CHECK-LABEL: func.func @udon_onnx_binding_enqueue_advance_sequence_state()
// CHECK-SAME: attributes {udoncl.export, udoncl.internal_export}
// CHECK: udonhl.address_of @__udon_onnx_sequence_state_bank
// CHECK: %[[ROWS_ADDRESS:.*]] = udonhl.address_of @__udon_onnx_sequence_state_advanced_rows
// CHECK-NEXT: "udonhl.store"(%c2_i32, %[[ROWS_ADDRESS]])
// CHECK: %[[AGE_ADDRESS:.*]] = udonhl.address_of @__udon_onnx_sequence_state_fill_age
// CHECK-NEXT: %[[AGE:.*]] = "udonhl.load"(%[[AGE_ADDRESS]])
// CHECK-NEXT: %[[KNOWN:.*]] = arith.cmpi ne, %[[AGE]], %c0_i32 : i32
// CHECK-NEXT: %[[SUM:.*]] = arith.addi %[[AGE]], %c2_i32 : i32
// CHECK-NEXT: %[[CAPPED:.*]] = arith.minui %[[SUM]], %c3_i32 : i32
// CHECK-NEXT: arith.select %[[KNOWN]], %[[CAPPED]], %[[AGE]] : i32
// CHECK: return %c0_i32 : i32

// The partial advance takes the rows, rejects more than a step with
// INVALID_VALUE after the busy (7) and state (12) checks, flips the bank
// like the plain advance and records the rows it committed. It is not a
// Udon export, and it issues no Device command.
// CHECK-LABEL: func.func @udon_onnx_binding_enqueue_advance_sequence_state_rows(
// CHECK-SAME: %[[REQUESTED:[^:]*]]: i32) -> i32 {
// CHECK-NOT: call @__vrc_opencl_
// CHECK: arith.cmpi ule, %[[REQUESTED]], %c2_i32 : i32
// CHECK-NEXT: cf.cond_br %{{.*}}, ^[[VALID:bb[0-9]+]], ^[[INVALID:bb[0-9]+]]
// CHECK: return %c7_i32 : i32
// CHECK: return %c12_i32 : i32
// CHECK: ^[[INVALID]]:
// CHECK-NEXT: return %c1_i32 : i32
// CHECK: %[[BANK_ADDRESS:.*]] = udonhl.address_of @__udon_onnx_sequence_state_bank
// CHECK: "udonhl.store"(%{{.*}}, %[[BANK_ADDRESS]])
// CHECK: %[[PARTIAL_ADDRESS:.*]] = udonhl.address_of @__udon_onnx_sequence_state_advanced_rows
// CHECK-NEXT: "udonhl.store"(%[[REQUESTED]], %[[PARTIAL_ADDRESS]])
// CHECK: arith.addi %{{.*}}, %[[REQUESTED]] : i32
// CHECK: arith.minui %{{.*}}, %c3_i32 : i32
// CHECK-NOT: call @__vrc_opencl_
// CHECK: return %c0_i32 : i32

// A run reads the committed rows once and passes them after the zero rows
// Z = carried - (age - 1) = 3 - age to both view kernels of each Concat.
// CHECK-LABEL: func.func @udononnxGeneratedEnqueueRun
// CHECK: udonhl.address_of @__udon_onnx_sequence_state_bank
// CHECK: %[[RUN_ROWS_ADDRESS:.*]] = udonhl.address_of @__udon_onnx_sequence_state_advanced_rows
// CHECK-NEXT: %[[RUN_ROWS:.*]] = "udonhl.load"(%[[RUN_ROWS_ADDRESS]])
// CHECK: %[[RUN_AGE:.*]] = "udonhl.load"
// CHECK: arith.subi %c3_i32, %[[RUN_AGE]] : i32
// CHECK: call @__vrc_opencl_dispatch_udononnx_fused_0(%[[KEY_PRESENT:[^,]*]], %[[TOKEN:[^,]*]], %[[KEY_PAST:[^,]*]], %[[ZERO_ROWS:[^,]*]], %[[RUN_ROWS]])
// CHECK: call @__vrc_opencl_dispatch_udononnx_fused_0(%[[KEY_PAST]], %[[TOKEN]], %[[KEY_PRESENT]], %[[ZERO_ROWS]], %[[RUN_ROWS]])
// CHECK: call @__vrc_opencl_dispatch_udononnx_fused_1(%[[KEY_PAST]], %[[TOKEN]], %[[KEY_PRESENT]], %[[ZERO_ROWS]])
// CHECK: call @__vrc_opencl_dispatch_udononnx_fused_2(%{{[^,]*}}, %[[TOKEN]], %{{[^,]*}}, %[[ZERO_ROWS]], %[[RUN_ROWS]])
// CHECK: call @__vrc_opencl_dispatch_udononnx_fused_2(%{{[^,]*}}, %[[TOKEN]], %{{[^,]*}}, %[[ZERO_ROWS]], %[[RUN_ROWS]])

// The view kernel bounds the run-time offset by the step: the carried row r
// of head h is row min(r, 1) + min(offset, 2) of head h in a bank of four
// rows, so every Load stays proven in bounds.
// CHECK-LABEL: gpu.module @udononnx_fused_kernels
// CHECK: gpu.func @udononnx_fused_0(%[[BANK_BUFFER:.*]]: memref<64xf32>, %{{.*}}: memref<32xf32>, %{{.*}}: memref<8192xf32>, %{{.*}}: i32, %[[OFFSET:.*]]: i32)
// CHECK: %[[BOUNDED_OFFSET:.*]] = arith.minui %[[OFFSET]], %c2_i32 : i32
// CHECK: %[[ROW:.*]] = arith.minui %{{.*}}, %c1_i32 : i32
// CHECK-NEXT: %[[MOVED:.*]] = arith.addi %[[ROW]], %[[BOUNDED_OFFSET]] : i32
// CHECK: memref.load %[[BANK_BUFFER]][%{{.*}}] : memref<64xf32>
// The bank-0 kernel reads the past tensor itself and takes no offset.
// CHECK: gpu.func @udononnx_fused_1(%{{.*}}: memref<32xf32>, %{{.*}}: memref<32xf32>, %{{.*}}: memref<8192xf32>, %{{.*}}: i32) kernel
// CHECK: gpu.func @udononnx_fused_2(%{{.*}}: memref<64xf32>, %{{.*}}: memref<32xf32>, %{{.*}}: memref<8192xf32>, %{{.*}}: i32, %{{.*}}: i32)

// PROVEN: gpu.module @device_gpu_4096 {
// PROVEN-NEXT: gpu.func @udononnx_fused_0({{.*}}vrc.gather_proven
// PROVEN: gpu.module @device_gpu_4097 {
// PROVEN-NEXT: gpu.func @udononnx_fused_1({{.*}}vrc.gather_proven
// PROVEN: gpu.module @device_gpu_4098 {
// PROVEN-NEXT: gpu.func @udononnx_fused_2({{.*}}vrc.gather_proven

// Without banks the partial advance copies the carried rows with a Slice that
// starts the requested rows into the present: the start word of axes 2 and 3
// is the run-time row count.
// COPIED-NOT: __udon_onnx_sequence_state_advanced_rows
// COPIED-LABEL: func.func @udon_onnx_binding_enqueue_advance_sequence_state()
// COPIED: call @__vrc_opencl_dispatch_udononnx_slice_f32({{.*}}, %c0_i32, %c2_i32, %c0_i32, %c65537_i32, %c65537_i32, %c65537_i32)
// COPIED-LABEL: func.func @udon_onnx_binding_enqueue_advance_sequence_state_rows(
// COPIED-SAME: %[[COPIED_ROWS:[^:]*]]: i32) -> i32
// COPIED-SAME: vrc.async_effect = "may_suspend"
// COPIED: call @__vrc_opencl_dispatch_udononnx_slice_f32({{.*}}, %c0_i32, %[[COPIED_ROWS]], %c0_i32, %c65537_i32, %c65537_i32, %c65537_i32)
// COPIED: call @__vrc_opencl_dispatch_udononnx_slice_f32({{.*}}, %c0_i32, %[[COPIED_ROWS]], %c0_i32, %c65537_i32, %c65537_i32, %c65537_i32)