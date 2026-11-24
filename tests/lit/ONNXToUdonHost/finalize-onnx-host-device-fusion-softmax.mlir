// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph %s | %FileCheck %s --check-prefix=PLAN
// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph --udoncl-materialize-onnx-host-globals --udoncl-materialize-onnx-host-model-registry --udoncl-attach-onnx-host-metadata --udoncl-finalize-onnx-host --udoncl-cleanup-onnx-host-source --canonicalize %s | %FileCheck %s
// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph --udoncl-materialize-onnx-host-globals --udoncl-materialize-onnx-host-model-registry --udoncl-attach-onnx-host-metadata --udoncl-finalize-onnx-host --udoncl-cleanup-onnx-host-source --canonicalize %s | %vrc-opt --vrc-assign-synthesized-kernel-ids=first-id=4096 --vrc-prove-synthesized-gather-kernels | %FileCheck %s --check-prefix=PROVEN
// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph=fuse-device-nodes=false --udoncl-materialize-onnx-host-globals --udoncl-materialize-onnx-host-model-registry --udoncl-attach-onnx-host-metadata --udoncl-finalize-onnx-host --udoncl-cleanup-onnx-host-source --canonicalize %s | %FileCheck %s --check-prefix=UNFUSED

// A Softmax is a reduction root: a per-element tree that only it reads, here
// the additive mask of attention scores, is evaluated inside its kernel at
// every element of the normalized axis instead of being materialized. An
// operand that another node also reads stays a buffer, and a Softmax is
// never itself interior.

func.func private @clEnqueueMarkerWithWaitList(
    !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32

func.func private @__udon_onnx_add(
    tensor<1x2x1x8xf32>, tensor<1x1x1x8xf32>) -> tensor<1x2x1x8xf32>
    attributes {
      udon.onnx.operation = "Add",
      udon.onnx.provider = "udonatenAdd"
    }
func.func private @__udon_onnx_softmax(
    tensor<1x2x1x8xf32>) -> tensor<1x2x1x8xf32>
    attributes {
      udon.onnx.operation = "Softmax",
      udon.onnx.provider = "udondnnSoftmaxForward"
    }
func.func private @__udon_onnx_mul(
    tensor<1x2x1x8xf32>, tensor<1x2x1x8xf32>) -> tensor<1x2x1x8xf32>
    attributes {
      udon.onnx.operation = "Mul",
      udon.onnx.provider = "udonatenMul"
    }

func.func @main_graph(
    %scores: tensor<1x2x1x8xf32> {onnx.name = "scores"},
    %mask: tensor<1x1x1x8xf32> {onnx.name = "mask"},
    %other: tensor<1x2x1x8xf32> {onnx.name = "other"})
    -> (tensor<1x2x1x8xf32> {onnx.name = "probabilities"},
        tensor<1x2x1x8xf32> {onnx.name = "shared"},
        tensor<1x2x1x8xf32> {onnx.name = "product"}) {
  // (1) softmax(scores + mask): one dispatch.
  %masked = func.call @__udon_onnx_add(%scores, %mask)
      : (tensor<1x2x1x8xf32>, tensor<1x1x1x8xf32>) -> tensor<1x2x1x8xf32>
  %probabilities = func.call @__udon_onnx_softmax(%masked)
      {udon.onnx.attributes = {axis = -1 : si64}}
      : (tensor<1x2x1x8xf32>) -> tensor<1x2x1x8xf32>
  // (2) The sum is also read by the Mul, so it keeps its own dispatch and
  // the Softmax reads its buffer.
  %sum = func.call @__udon_onnx_add(%other, %mask)
      : (tensor<1x2x1x8xf32>, tensor<1x1x1x8xf32>) -> tensor<1x2x1x8xf32>
  %shared = func.call @__udon_onnx_softmax(%sum)
      {udon.onnx.attributes = {axis = -1 : si64}}
      : (tensor<1x2x1x8xf32>) -> tensor<1x2x1x8xf32>
  // (3) A Softmax result fuses into nothing: the Mul reads it as a leaf.
  %product = func.call @__udon_onnx_mul(%shared, %sum)
      : (tensor<1x2x1x8xf32>, tensor<1x2x1x8xf32>) -> tensor<1x2x1x8xf32>
  return %probabilities, %shared, %product
      : tensor<1x2x1x8xf32>, tensor<1x2x1x8xf32>, tensor<1x2x1x8xf32>
}

// PLAN: device_fusion_roots = array<i64: 1, 1, 2, 3, 4>

// CHECK-LABEL: func.func @udononnxGeneratedRun(
// CHECK: call @__vrc_opencl_dispatch_udononnx_fused_0(%[[SCORES:[^,]*]], %[[MASK:[^,]*]], %{{[^,]*}}) : (i32, i32, i32) -> i32
// CHECK: call @__vrc_opencl_dispatch_udondnn_large_op_tensor
// CHECK: call @__vrc_opencl_dispatch_udondnn_arena_softmax
// CHECK: call @__vrc_opencl_dispatch_udondnn_large_op_tensor
// CHECK-NOT: call @__vrc_opencl_dispatch_

// The kernel repeats udondnn_arena_softmax_forward over the last axis with
// the Add in place of the input Load: a maximum from -FLT_MAX over samples
// that compare greater, an ascending sum of exp(sample - maximum), and
// exp(own sample - maximum) / sum. Each pass is a counted loop over the 8
// elements of the row, with the row base computed once.
// CHECK-LABEL: gpu.module @udononnx_fused_kernels {
// CHECK: gpu.func @udononnx_fused_0(%[[S:.*]]: memref<16xf32>, %[[M:.*]]: memref<8xf32>, %{{.*}}: memref<8192xf32>) kernel
// CHECK-SAME: vrc.fragment_cost = {loads = 34 : i64}
// CHECK-DAG: %[[LOWEST:.*]] = arith.constant -3.40282347E+38 : f32
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[C8:.*]] = arith.constant 8 : i32
// CHECK: %[[I:.*]] = arith.index_cast %{{.*}} : index to i32
// CHECK: scf.if
// CHECK: %[[MAXIMUM:.*]]:2 = scf.while (%[[K:.*]] = %{{.*}}, %[[BEST:.*]] = %[[LOWEST]]) : (i32, f32) -> (i32, f32) {
// CHECK:   %[[MORE:.*]] = arith.cmpi ult, %[[K]], %[[C8]] : i32
// CHECK:   scf.condition(%[[MORE]]) %[[K]], %[[BEST]] : i32, f32
// CHECK: } do {
// CHECK: ^bb0(%[[J:.*]]: i32, %[[CURRENT:.*]]: f32):
// CHECK:   %[[SCORE:.*]] = memref.load %[[S]][
// CHECK:   %[[MASK_VALUE:.*]] = memref.load %[[M]][
// CHECK:   %[[SAMPLE:.*]] = arith.addf %[[SCORE]], %[[MASK_VALUE]] : f32
// CHECK:   %[[GREATER:.*]] = arith.cmpf ogt, %[[SAMPLE]], %[[CURRENT]] : f32
// CHECK:   %[[KEPT:.*]] = arith.select %[[GREATER]], %[[SAMPLE]], %[[CURRENT]] : f32
// CHECK:   scf.yield %{{.*}}, %[[KEPT]] : i32, f32
// CHECK: }
// CHECK: %[[SUM:.*]]:2 = scf.while (%{{.*}} = %{{.*}}, %{{.*}} = %[[ZERO]]) : (i32, f32) -> (i32, f32) {
// CHECK: } do {
// CHECK: ^bb0(%{{.*}}: i32, %[[TOTAL:.*]]: f32):
// CHECK:   %[[SCORE2:.*]] = memref.load %[[S]][
// CHECK:   %[[MASK2:.*]] = memref.load %[[M]][
// CHECK:   %[[SAMPLE2:.*]] = arith.addf %[[SCORE2]], %[[MASK2]] : f32
// CHECK:   %[[SHIFTED:.*]] = arith.subf %[[SAMPLE2]], %[[MAXIMUM]]#1 : f32
// CHECK:   %[[EXP:.*]] = math.exp %[[SHIFTED]] : f32
// CHECK:   %[[NEXT:.*]] = arith.addf %[[TOTAL]], %[[EXP]] : f32
// CHECK:   scf.yield %{{.*}}, %[[NEXT]] : i32, f32
// CHECK: }
// CHECK: %[[OWN_SCORE:.*]] = memref.load %[[S]][
// CHECK: %[[OWN_MASK:.*]] = memref.load %[[M]][
// CHECK: %[[OWN:.*]] = arith.addf %[[OWN_SCORE]], %[[OWN_MASK]] : f32
// CHECK: %[[OWN_SHIFTED:.*]] = arith.subf %[[OWN]], %[[MAXIMUM]]#1 : f32
// CHECK: %[[OWN_EXP:.*]] = math.exp %[[OWN_SHIFTED]] : f32
// CHECK: arith.divf %[[OWN_EXP]], %[[SUM]]#1 : f32
// CHECK-NOT: gpu.func

// PROVEN: gpu.module @device_gpu_4096 {
// PROVEN-NEXT: gpu.func @udononnx_fused_0({{.*}}vrc.gather_proven, vrc.kernel_id = 4096 : i32
// PROVEN-NOT: gpu.module

// UNFUSED-LABEL: func.func @udononnxGeneratedRun(
// UNFUSED: call @__vrc_opencl_dispatch_udondnn_large_op_tensor
// UNFUSED: call @__vrc_opencl_dispatch_udondnn_arena_softmax
// UNFUSED: call @__vrc_opencl_dispatch_udondnn_large_op_tensor
// UNFUSED: call @__vrc_opencl_dispatch_udondnn_arena_softmax
// UNFUSED: call @__vrc_opencl_dispatch_udondnn_large_op_tensor
// UNFUSED-NOT: call @__vrc_opencl_dispatch_
// UNFUSED-NOT: gpu.module
