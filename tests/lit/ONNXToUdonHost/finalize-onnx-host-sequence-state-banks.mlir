// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph %s | %FileCheck %s --check-prefix=PLAN
// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph --udoncl-materialize-onnx-host-globals --udoncl-materialize-onnx-host-model-registry --udoncl-attach-onnx-host-metadata --udoncl-finalize-onnx-host --udoncl-cleanup-onnx-host-source --canonicalize %s | %FileCheck %s
// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph --udoncl-materialize-onnx-host-globals --udoncl-materialize-onnx-host-model-registry --udoncl-attach-onnx-host-metadata --udoncl-finalize-onnx-host --udoncl-cleanup-onnx-host-source --canonicalize %s | %vrc-opt --vrc-assign-synthesized-kernel-ids=first-id=4096 --vrc-prove-synthesized-gather-kernels | %FileCheck %s --check-prefix=PROVEN
// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph=bank-sequence-state=false --udoncl-materialize-onnx-host-globals --udoncl-materialize-onnx-host-model-registry --udoncl-attach-onnx-host-metadata --udoncl-finalize-onnx-host --udoncl-cleanup-onnx-host-source --canonicalize %s | %FileCheck %s --check-prefix=COPIED

// Two sequence states of capacity 4 that advance by one row. Each run
// writes its present into one of two banks, the present output or the
// present-sized past input, and reads the carried rows from the other bank
// through a view, so the advance copies nothing. The key Concat is the root
// of a fused group; the value Concat is a library dispatch on its own.

module attributes {udoncl.onnx_sequence_states = [
    {axis = 2 : i64, capacity = 4 : i64, input_index = 1 : i64,
     output_index = 0 : i64, step = 1 : i64},
    {axis = 2 : i64, capacity = 4 : i64, input_index = 2 : i64,
     output_index = 1 : i64, step = 1 : i64}]} {
  func.func private @clEnqueueMarkerWithWaitList(
      !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32

  func.func private @__udon_onnx_neg(tensor<1x2x1x8xf32>)
      -> tensor<1x2x1x8xf32>
      attributes {
        udon.onnx.operation = "Neg",
        udon.onnx.provider = "udononnxNeg"
      }
  func.func private @__udon_onnx_concat(
      tensor<1x2x3x8xf32>, tensor<1x2x1x8xf32>) -> tensor<1x2x4x8xf32>
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
      tensor<1x2x1x4xf32>, tensor<1x2x4x8xf32>) -> tensor<1x2x1x8xf32>
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
      %token: tensor<1x2x1x8xf32> {onnx.name = "token"},
      %past_key: tensor<1x2x3x8xf32> {onnx.name = "past_key"},
      %past_value: tensor<1x2x3x8xf32> {onnx.name = "past_value"},
      %weights: tensor<1x2x1x4xf32> {onnx.name = "weights"})
      -> (tensor<1x2x4x8xf32> {onnx.name = "present_key"},
          tensor<1x2x4x8xf32> {onnx.name = "present_value"},
          tensor<1x2x4x8xf32> {onnx.name = "scaled_keys"},
          tensor<1x2x1x8xf32> {onnx.name = "mixed"}) {
    %scale = func.call @__static_scale() : () -> tensor<f32>
    %negated = func.call @__udon_onnx_neg(%token)
        : (tensor<1x2x1x8xf32>) -> tensor<1x2x1x8xf32>
    %present_key = func.call @__udon_onnx_concat(%past_key, %negated)
        {udon.onnx.attributes = {axis = 2 : si64}}
        : (tensor<1x2x3x8xf32>, tensor<1x2x1x8xf32>) -> tensor<1x2x4x8xf32>
    %present_value = func.call @__udon_onnx_concat(%past_value, %token)
        {udon.onnx.attributes = {axis = -2 : si64}}
        : (tensor<1x2x3x8xf32>, tensor<1x2x1x8xf32>) -> tensor<1x2x4x8xf32>
    %scaled = func.call @__udon_onnx_mul_scalar(%present_key, %scale)
        : (tensor<1x2x4x8xf32>, tensor<f32>) -> tensor<1x2x4x8xf32>
    %mixed = func.call @__udon_onnx_matmul(%weights, %present_value)
        : (tensor<1x2x1x4xf32>, tensor<1x2x4x8xf32>) -> tensor<1x2x1x8xf32>
    return %present_key, %present_value, %scaled, %mixed
        : tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32>,
          tensor<1x2x1x8xf32>
  }
}

// Both states are banked. A past input holds a whole present: 256 bytes
// instead of the 192 of the past tensor.
// PLAN: udoncl.onnx_host_plan = {
// PLAN-SAME: mutable_buffer_bytes = array<i64: 64, 256, 256, 32, 256, 256, 256, 64>
// PLAN-SAME: sequence_state_banks = array<i64: 0, 1>

// create and bind-all return the carried rows to the past input.
// CHECK-LABEL: func.func @__udononnx_transitional_create
// CHECK: udonhl.address_of @__udon_onnx_sequence_state_bank
// CHECK-LABEL: func.func @__udononnx_transitional_bind_all
// CHECK: udonhl.address_of @__udon_onnx_sequence_state_bank

// Reset clears the past inputs and returns to bank 0.
// CHECK-LABEL: func.func @__udononnx_transitional_reset_sequence_state
// CHECK: call @__vrc_opencl_buffer_fill
// CHECK: call @__vrc_opencl_buffer_fill
// CHECK: %[[RESET_BANK:.*]] = udonhl.address_of @__udon_onnx_sequence_state_bank
// CHECK-NEXT: "udonhl.store"(%c0_i32, %[[RESET_BANK]])

// The advance only flips the bank: 2 after a run from bank 1, 1 after a run
// from bank 0 or 2.
// CHECK-LABEL: func.func @__udononnx_transitional_advance_sequence_state
// CHECK-NOT: call @__vrc_opencl_
// CHECK: %[[BANK_ADDRESS:.*]] = udonhl.address_of @__udon_onnx_sequence_state_bank
// CHECK: %[[BANK:.*]] = "udonhl.load"(%[[BANK_ADDRESS]])
// CHECK: %[[FROM_INPUT:.*]] = arith.cmpi eq, %[[BANK]], %c1_i32 : i32
// CHECK: %[[NEXT:.*]] = arith.select %[[FROM_INPUT]], %c2_i32, %c1_i32 : i32
// CHECK: "udonhl.store"(%[[NEXT]], %[[BANK_ADDRESS]])
// CHECK-NOT: call @__vrc_opencl_
// CHECK: return %c0_i32 : i32

// Materializing copies each present from its past input (logical buffers 2
// and 3) into its present output (5 and 6) only when a run from bank 1 left
// it there.
// CHECK-LABEL: func.func @__udononnx_transitional_materialize_sequence_state
// CHECK: %[[KEY_BANK:.*]] = call @__vrc_opencl_buffer_alloc(%c2_i32, %c256_i32)
// CHECK: %[[KEY_OUTPUT:.*]] = call @__vrc_opencl_buffer_alloc(%c5_i32, %c256_i32)
// CHECK: call @__vrc_opencl_buffer_copy(%[[KEY_BANK]], %[[KEY_OUTPUT]], %c0_i32, %c0_i32, %c256_i32)
// CHECK: %[[VALUE_BANK:.*]] = call @__vrc_opencl_buffer_alloc(%c3_i32, %c256_i32)
// CHECK: %[[VALUE_OUTPUT:.*]] = call @__vrc_opencl_buffer_alloc(%c6_i32, %c256_i32)
// CHECK: call @__vrc_opencl_buffer_copy(%[[VALUE_BANK]], %[[VALUE_OUTPUT]], %c0_i32, %c0_i32, %c256_i32)
// CHECK: udonhl.address_of @__udon_onnx_sequence_state_bank

// The run reads the bank once. The key Concat runs its view kernel from
// the present output into the past input in bank 1, from the past input
// into the present output in bank 2, and its own fused kernel in bank 0;
// the value Concat likewise, with the library Concat in bank 0. A reader
// of a present reads the past input in bank 1.
// CHECK-LABEL: func.func @udononnxGeneratedEnqueueRun
// CHECK: %[[RUN_BANK_ADDRESS:.*]] = udonhl.address_of @__udon_onnx_sequence_state_bank
// CHECK: %[[RUN_BANK:.*]] = "udonhl.load"(%[[RUN_BANK_ADDRESS]])
// CHECK: arith.cmpi eq, %[[RUN_BANK]], %c1_i32 : i32
// The Concats also take the zero rows of the states
// (finalize-onnx-host-sequence-zero-rows.mlir).
// CHECK: call @__vrc_opencl_dispatch_udononnx_fused_0(%[[KEY_PRESENT:[^,]*]], %[[TOKEN:[^,]*]], %[[KEY_PAST:[^,]*]], %[[ZERO_ROWS:[^)]*]])
// CHECK: arith.cmpi eq, %[[RUN_BANK]], %c2_i32 : i32
// CHECK: call @__vrc_opencl_dispatch_udononnx_fused_0(%[[KEY_PAST]], %[[TOKEN]], %[[KEY_PRESENT]], %[[ZERO_ROWS]])
// CHECK: call @__vrc_opencl_dispatch_udononnx_fused_1(%[[KEY_PAST]], %[[TOKEN]], %[[KEY_PRESENT]], %[[ZERO_ROWS]])
// CHECK: call @__vrc_opencl_dispatch_udononnx_fused_2(%[[VALUE_PRESENT:[^,]*]], %[[TOKEN]], %[[VALUE_PAST:[^,]*]], %[[ZERO_ROWS]])
// CHECK: call @__vrc_opencl_dispatch_udononnx_fused_2(%[[VALUE_PAST]], %[[TOKEN]], %[[VALUE_PRESENT]], %[[ZERO_ROWS]])
// CHECK: call @__vrc_opencl_dispatch_udononnx_concat2_f32(%[[VALUE_PAST]], %[[TOKEN]], %[[VALUE_PRESENT]],
// CHECK: call @__vrc_opencl_dispatch_udondnn_large_op_tensor(%[[KEY_PAST]],
// CHECK: call @__vrc_opencl_dispatch_udondnn_large_op_tensor(%[[KEY_PRESENT]],
// CHECK: call @__vrc_opencl_dispatch_udonblas_arena_sgemm_strided_batched(%{{[^,]*}}, %[[VALUE_PAST]],
// CHECK: call @__vrc_opencl_dispatch_udonblas_arena_sgemm_strided_batched(%{{[^,]*}}, %[[VALUE_PRESENT]],
// CHECK-NOT: call @__vrc_opencl_dispatch_
// CHECK: return %c0_i32 : i32

// The view kernel of the key: the carried row r of head h is row r + 1 of
// head h in a bank that holds a whole present.
// CHECK-LABEL: gpu.module @udononnx_fused_kernels
// CHECK: gpu.func @udononnx_fused_0(%[[BANK_BUFFER:.*]]: memref<64xf32>, %[[NEW:.*]]: memref<16xf32>, %{{.*}}: memref<8192xf32>, %{{.*}}: i32)
// CHECK:   %[[HEAD:.*]] = arith.divui %{{.*}}, %c32_i32 : i32
// CHECK:   %[[ROW:.*]] = arith.remui %{{.*}}, %c4_i32 : i32
// CHECK:   %[[COLUMN:.*]] = arith.remui %{{.*}}, %c8_i32 : i32
// CHECK:   %[[CARRIED:.*]] = arith.cmpi ult, %[[ROW]], %c3_i32 : i32
// CHECK:   scf.if %[[CARRIED]] -> (f32) {
// CHECK:     %[[BOUNDED:.*]] = arith.minui %[[ROW]], %c2_i32 : i32
// CHECK:     %[[MOVED:.*]] = arith.addi %[[BOUNDED]], %c1_i32 : i32
// CHECK:     %[[HEAD_ROWS:.*]] = arith.muli %[[HEAD]], %c4_i32 : i32
// CHECK:     %[[BANK_ROW:.*]] = arith.addi %[[HEAD_ROWS]], %[[MOVED]] : i32
// CHECK:     %[[BANK_BASE:.*]] = arith.muli %[[BANK_ROW]], %c8_i32 : i32
// CHECK:     %[[BANK_LINEAR:.*]] = arith.addi %[[BANK_BASE]], %[[COLUMN]] : i32
// CHECK:     %[[BANK_INDEX:.*]] = arith.index_cast %[[BANK_LINEAR]] : i32 to index
// CHECK:     memref.load %[[BANK_BUFFER]][%[[BANK_INDEX]]] : memref<64xf32>
// CHECK:   } else {
// CHECK:     %[[NEW_VALUE:.*]] = memref.load %[[NEW]][%{{.*}}] : memref<16xf32>
// CHECK:     arith.negf %[[NEW_VALUE]] : f32
// The fused kernel of bank 0 reads the 48-element past tensor itself.
// CHECK: gpu.func @udononnx_fused_1(%{{.*}}: memref<48xf32>, %{{.*}}: memref<16xf32>, %{{.*}}: memref<8192xf32>, %{{.*}}: i32)
// CHECK: gpu.func @udononnx_fused_2(%{{.*}}: memref<64xf32>, %{{.*}}: memref<16xf32>, %{{.*}}: memref<8192xf32>, %{{.*}}: i32)
// CHECK-NOT: gpu.func

// PROVEN: gpu.module @device_gpu_4096 {
// PROVEN-NEXT: gpu.func @udononnx_fused_0({{.*}}vrc.gather_proven
// PROVEN: gpu.module @device_gpu_4097 {
// PROVEN-NEXT: gpu.func @udononnx_fused_1({{.*}}vrc.gather_proven
// PROVEN: gpu.module @device_gpu_4098 {
// PROVEN-NEXT: gpu.func @udononnx_fused_2({{.*}}vrc.gather_proven

// Without banks every advance copies the carried rows with a Slice, and
// nothing needs materializing.
// COPIED-NOT: __udon_onnx_sequence_state_bank
// COPIED-LABEL: func.func @__udononnx_transitional_advance_sequence_state
// COPIED: call @__vrc_opencl_dispatch_udononnx_slice_f32
// COPIED: call @__vrc_opencl_dispatch_udononnx_slice_f32
// COPIED-NOT: __udononnx_transitional_materialize_sequence_state
// COPIED-LABEL: func.func @udononnxGeneratedEnqueueRun
// COPIED: call @__vrc_opencl_dispatch_udononnx_fused_0(
// COPIED: call @__vrc_opencl_dispatch_udononnx_concat2_f32(
// COPIED: call @__vrc_opencl_dispatch_udondnn_large_op_tensor(
// COPIED: call @__vrc_opencl_dispatch_udonblas_arena_sgemm_strided_batched(
// COPIED-NOT: call @__vrc_opencl_dispatch_
