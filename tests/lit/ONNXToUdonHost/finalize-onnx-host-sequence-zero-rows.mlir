// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph --udoncl-materialize-onnx-host-globals --udoncl-materialize-onnx-host-model-registry --udoncl-attach-onnx-host-metadata --udoncl-finalize-onnx-host --udoncl-cleanup-onnx-host-source --canonicalize %s | %FileCheck %s
// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph --udoncl-materialize-onnx-host-globals --udoncl-materialize-onnx-host-model-registry --udoncl-attach-onnx-host-metadata --udoncl-finalize-onnx-host --udoncl-cleanup-onnx-host-source --canonicalize %s | %vrc-opt --vrc-assign-synthesized-kernel-ids=first-id=4096 --vrc-prove-synthesized-gather-kernels | %FileCheck %s --check-prefix=PROVEN

// The attention of one decoder layer over two sequence states of capacity
// 16 that advance by one row. After the binding's reset the first rows of
// every past input hold its zero fill; the binding counts the advances
// since (the fill age), and a run derives the zero rows Z = 15 - advances
// of both states (ONNXHostSequenceExtents.h):
// - both Concats write their first Z rows as +0 without reading the past;
// - the scaled transposed keys write their first Z columns as +0 * 0.5;
// - the masked Softmax writes its first Z columns as +0, because the only
//   reader of its result, the value product, skips those terms;
// - the value product passes Z as k_begin;
// - the score product reads every key column, and a product whose left
//   operand is not a Softmax result (`mixed`) skips nothing.

module attributes {udoncl.onnx_sequence_states = [
    {axis = 2 : i64, capacity = 16 : i64, input_index = 1 : i64,
     output_index = 0 : i64, step = 1 : i64},
    {axis = 2 : i64, capacity = 16 : i64, input_index = 2 : i64,
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
      tensor<1x2x15x8xf32>, tensor<1x2x1x8xf32>) -> tensor<1x2x16x8xf32>
      attributes {
        udon.onnx.operation = "Concat",
        udon.onnx.provider = "udononnxConcat"
      }
  func.func private @__udon_onnx_transpose(tensor<1x2x16x8xf32>)
      -> tensor<1x2x8x16xf32>
      attributes {
        udon.onnx.operation = "Transpose",
        udon.onnx.provider = "udononnxTranspose"
      }
  func.func private @__udon_onnx_mul_scalar(
      tensor<1x2x8x16xf32>, tensor<f32>) -> tensor<1x2x8x16xf32>
      attributes {
        udon.onnx.operation = "Mul",
        udon.onnx.provider = "udonatenMul"
      }
  func.func private @__udon_onnx_scores(
      tensor<1x2x1x8xf32>, tensor<1x2x8x16xf32>) -> tensor<1x2x1x16xf32>
      attributes {
        udon.onnx.operation = "MatMul",
        udon.onnx.provider = "udonblasSgemm"
      }
  func.func private @__udon_onnx_add(
      tensor<1x2x1x16xf32>, tensor<1x1x1x16xf32>) -> tensor<1x2x1x16xf32>
      attributes {
        udon.onnx.operation = "Add",
        udon.onnx.provider = "udonatenAdd"
      }
  func.func private @__udon_onnx_softmax(tensor<1x2x1x16xf32>)
      -> tensor<1x2x1x16xf32>
      attributes {
        udon.onnx.operation = "Softmax",
        udon.onnx.provider = "udondnnSoftmaxForward"
      }
  func.func private @__udon_onnx_values(
      tensor<1x2x1x16xf32>, tensor<1x2x16x8xf32>) -> tensor<1x2x1x8xf32>
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
      %past_key: tensor<1x2x15x8xf32> {onnx.name = "past_key"},
      %past_value: tensor<1x2x15x8xf32> {onnx.name = "past_value"},
      %query: tensor<1x2x1x8xf32> {onnx.name = "query"},
      %mask: tensor<1x1x1x16xf32> {onnx.name = "mask"},
      %weights: tensor<1x2x1x16xf32> {onnx.name = "weights"})
      -> (tensor<1x2x16x8xf32> {onnx.name = "present_key"},
          tensor<1x2x16x8xf32> {onnx.name = "present_value"},
          tensor<1x2x1x8xf32> {onnx.name = "attended"},
          tensor<1x2x1x8xf32> {onnx.name = "mixed"}) {
    %scale = func.call @__static_scale() : () -> tensor<f32>
    %negated = func.call @__udon_onnx_neg(%token)
        : (tensor<1x2x1x8xf32>) -> tensor<1x2x1x8xf32>
    %present_key = func.call @__udon_onnx_concat(%past_key, %negated)
        {udon.onnx.attributes = {axis = 2 : si64}}
        : (tensor<1x2x15x8xf32>, tensor<1x2x1x8xf32>) -> tensor<1x2x16x8xf32>
    %present_value = func.call @__udon_onnx_concat(%past_value, %token)
        {udon.onnx.attributes = {axis = 2 : si64}}
        : (tensor<1x2x15x8xf32>, tensor<1x2x1x8xf32>) -> tensor<1x2x16x8xf32>
    %keys = func.call @__udon_onnx_transpose(%present_key)
        {udon.onnx.attributes = {perm = [0, 1, 3, 2]}}
        : (tensor<1x2x16x8xf32>) -> tensor<1x2x8x16xf32>
    %scaled = func.call @__udon_onnx_mul_scalar(%keys, %scale)
        : (tensor<1x2x8x16xf32>, tensor<f32>) -> tensor<1x2x8x16xf32>
    %scores = func.call @__udon_onnx_scores(%query, %scaled)
        : (tensor<1x2x1x8xf32>, tensor<1x2x8x16xf32>) -> tensor<1x2x1x16xf32>
    %masked = func.call @__udon_onnx_add(%scores, %mask)
        : (tensor<1x2x1x16xf32>, tensor<1x1x1x16xf32>) -> tensor<1x2x1x16xf32>
    %probabilities = func.call @__udon_onnx_softmax(%masked)
        {udon.onnx.attributes = {axis = -1 : si64}}
        : (tensor<1x2x1x16xf32>) -> tensor<1x2x1x16xf32>
    %attended = func.call @__udon_onnx_values(%probabilities, %present_value)
        : (tensor<1x2x1x16xf32>, tensor<1x2x16x8xf32>) -> tensor<1x2x1x8xf32>
    %mixed = func.call @__udon_onnx_values(%weights, %present_value)
        : (tensor<1x2x1x16xf32>, tensor<1x2x16x8xf32>) -> tensor<1x2x1x8xf32>
    return %present_key, %present_value, %attended, %mixed
        : tensor<1x2x16x8xf32>, tensor<1x2x16x8xf32>, tensor<1x2x1x8xf32>,
          tensor<1x2x1x8xf32>
  }
}

// CHECK-DAG: sym_name = "__udon_onnx_sequence_state_fill_age"
// CHECK-DAG: sym_name = "__udon_onnx_sequence_state_exposed"

// create: nothing is filled and nobody holds a past input.
// CHECK-LABEL: func.func @__udononnx_transitional_create
// CHECK: %[[CREATE_AGE:.*]] = udonhl.address_of @__udon_onnx_sequence_state_fill_age
// CHECK-NEXT: "udonhl.store"(%c0_i32, %[[CREATE_AGE]])
// CHECK: %[[CREATE_EXPOSED:.*]] = udonhl.address_of @__udon_onnx_sequence_state_exposed
// CHECK-NEXT: "udonhl.store"(%c0_i32, %[[CREATE_EXPOSED]])

// bind-all forgets the fill.
// CHECK-LABEL: func.func @__udononnx_transitional_bind_all
// CHECK: %[[BIND_AGE:.*]] = udonhl.address_of @__udon_onnx_sequence_state_fill_age
// CHECK-NEXT: "udonhl.store"(%c0_i32, %[[BIND_AGE]])

// Resolving input 1 or 2 (a past input) exposes the states for good.
// CHECK-LABEL: func.func @__udononnx_transitional_resolve_input
// CHECK: udonhl.address_of @__udon_onnx_sequence_state_exposed
// CHECK: arith.cmpi eq, %arg0, %c1_i32
// CHECK: arith.cmpi eq, %arg0, %c2_i32
// CHECK: udonhl.address_of @__udon_onnx_sequence_state_fill_age
// CHECK: call @udononnxBindingStateResolveInput
// CHECK-LABEL: func.func @__udononnx_transitional_resolve_output
// CHECK-NOT: sequence_state_exposed
// CHECK: call @udononnxBindingStateResolveOutput

// Reset fills the past inputs; the age becomes 1 unless they are exposed.
// CHECK-LABEL: func.func @__udononnx_transitional_reset_sequence_state
// CHECK: call @__vrc_opencl_buffer_fill
// CHECK: call @__vrc_opencl_buffer_fill
// CHECK: %[[EXPOSED:.*]] = "udonhl.load"(%{{.*}}) {{.*}} -> i32
// CHECK-NEXT: %[[HIDDEN:.*]] = arith.cmpi eq, %[[EXPOSED]], %c0_i32
// CHECK-NEXT: %[[AGE:.*]] = arith.extui %[[HIDDEN]] : i1 to i32
// CHECK-NEXT: %[[RESET_AGE:.*]] = udonhl.address_of @__udon_onnx_sequence_state_fill_age
// CHECK-NEXT: "udonhl.store"(%[[AGE]], %[[RESET_AGE]])

// Each advance counts, up to 16 (15 carried rows, step 1, plus one).
// CHECK-LABEL: func.func @__udononnx_transitional_advance_sequence_state
// CHECK: %[[OLD_ADDRESS:.*]] = udonhl.address_of @__udon_onnx_sequence_state_fill_age
// CHECK-NEXT: %[[OLD:.*]] = "udonhl.load"(%[[OLD_ADDRESS]])
// CHECK-DAG: arith.cmpi slt, %[[OLD]], %c16_i32
// CHECK-DAG: arith.cmpi ne, %[[OLD]], %c0_i32
// CHECK: arith.addi %[[OLD]], %c1_i32

// The run reads the age once: Z = 16 - age when the age is known.
// CHECK-LABEL: func.func @udononnxGeneratedEnqueueRun
// CHECK: %[[RUN_AGE_ADDRESS:.*]] = udonhl.address_of @__udon_onnx_sequence_state_fill_age
// CHECK-NEXT: %[[RUN_AGE:.*]] = "udonhl.load"(%[[RUN_AGE_ADDRESS]])
// CHECK-NEXT: %[[KNOWN:.*]] = arith.cmpi ne, %[[RUN_AGE]], %c0_i32
// CHECK-NEXT: %[[ROWS:.*]] = arith.subi %c16_i32, %[[RUN_AGE]]
// CHECK-NEXT: %[[POSITIVE:.*]] = arith.cmpi sgt, %[[ROWS]], %c0_i32
// CHECK-NEXT: %[[BOTH:.*]] = arith.andi %[[KNOWN]], %[[POSITIVE]]
// CHECK-NEXT: %[[Z:.*]] = arith.select %[[BOTH]], %[[ROWS]], %c0_i32
// Both Concats in every bank, then the scaled keys, take Z.
// CHECK: call @__vrc_opencl_dispatch_udononnx_fused_{{[0-9]+}}({{.*}}, %[[Z]])
// The score product reads every key column.
// CHECK: call @__vrc_opencl_dispatch_udonblas_arena_sgemm_strided_batched({{.*}}, %cst{{.*}}, %c0_i32)
// The masked Softmax writes its first Z columns as +0.
// CHECK: call @__vrc_opencl_dispatch_udononnx_fused_{{[0-9]+}}({{.*}}, %[[Z]])
// The value product skips the first Z terms in every bank; `mixed` skips none.
// CHECK: call @__vrc_opencl_dispatch_udonblas_arena_sgemm_strided_batched({{.*}}, %[[Z]])
// CHECK: call @__vrc_opencl_dispatch_udonblas_arena_sgemm_strided_batched({{.*}}, %c0_i32)

// The kernels take Z after the result buffer and write the constant without
// evaluating their expression below it.
// CHECK-LABEL: gpu.module @udononnx_fused_kernels
// CHECK: gpu.func @udononnx_fused_{{[0-9]+}}({{.*}}, %[[ROWS_ARG:.*]]: i32) kernel
// CHECK: arith.cmpi ult, %{{.*}}, %[[ROWS_ARG]] : i32
// CHECK-NEXT: scf.if

// The synthesized kernels keep their bounds proof with the scalar.
// PROVEN-COUNT-5: vrc.gather_proven
