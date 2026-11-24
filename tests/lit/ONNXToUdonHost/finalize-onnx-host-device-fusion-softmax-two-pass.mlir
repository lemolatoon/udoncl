// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph %s | %FileCheck %s --check-prefix=PLAN
// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph --udoncl-materialize-onnx-host-globals --udoncl-materialize-onnx-host-model-registry --udoncl-attach-onnx-host-metadata --udoncl-finalize-onnx-host --udoncl-cleanup-onnx-host-source --canonicalize %s | %FileCheck %s
// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph --udoncl-materialize-onnx-host-globals --udoncl-materialize-onnx-host-model-registry --udoncl-attach-onnx-host-metadata --udoncl-finalize-onnx-host --udoncl-cleanup-onnx-host-source --canonicalize %s | %vrc-opt --vrc-assign-synthesized-kernel-ids=first-id=4096 --vrc-prove-synthesized-gather-kernels | %FileCheck %s --check-prefix=PROVEN

// A Softmax over rows of 512 runs in two passes (deviceSoftmaxSchedule): the
// one-pass kernel re-reduces the whole row in every output fragment, which
// grows with the square of the row. A statistics kernel writes, per row and
// chunk of 19 elements, the chunk maximum and the sum of exp(sample - chunk
// maximum) to the operator scratch (27 chunks: 864 words per plane), and the
// elementwise kernel combines its row's chunks. The fused mask Add is
// evaluated in both kernels; a Softmax outside every group runs the same way
// with its input as the only leaf.

func.func private @clEnqueueMarkerWithWaitList(
    !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32

func.func private @__udon_onnx_add(
    tensor<1x32x1x512xf32>, tensor<1x1x1x512xf32>) -> tensor<1x32x1x512xf32>
    attributes {
      udon.onnx.operation = "Add",
      udon.onnx.provider = "udonatenAdd"
    }
func.func private @__udon_onnx_softmax(
    tensor<1x32x1x512xf32>) -> tensor<1x32x1x512xf32>
    attributes {
      udon.onnx.operation = "Softmax",
      udon.onnx.provider = "udondnnSoftmaxForward"
    }

func.func @main_graph(
    %scores: tensor<1x32x1x512xf32> {onnx.name = "scores"},
    %mask: tensor<1x1x1x512xf32> {onnx.name = "mask"})
    -> (tensor<1x32x1x512xf32> {onnx.name = "probabilities"},
        tensor<1x32x1x512xf32> {onnx.name = "plain"}) {
  %masked = func.call @__udon_onnx_add(%scores, %mask)
      : (tensor<1x32x1x512xf32>, tensor<1x1x1x512xf32>)
      -> tensor<1x32x1x512xf32>
  %probabilities = func.call @__udon_onnx_softmax(%masked)
      {udon.onnx.attributes = {axis = -1 : si64}}
      : (tensor<1x32x1x512xf32>) -> tensor<1x32x1x512xf32>
  %plain = func.call @__udon_onnx_softmax(%scores)
      {udon.onnx.attributes = {axis = -1 : si64}}
      : (tensor<1x32x1x512xf32>) -> tensor<1x32x1x512xf32>
  return %probabilities, %plain
      : tensor<1x32x1x512xf32>, tensor<1x32x1x512xf32>
}

// PLAN: device_fusion_roots = array<i64: 1, 1, 2>
// PLAN-SAME: operator_scratch_bytes = 65536

// CHECK-LABEL: func.func @udononnxGeneratedRun(
// CHECK: call @__vrc_opencl_dispatch_udononnx_fused_0(%[[SCORES:[^,]*]], %[[MASK:[^,]*]], %[[SCRATCH:[^)]*]]) : (i32, i32, i32) -> i32
// CHECK: call @__vrc_opencl_dispatch_udononnx_fused_1(%[[SCORES]], %[[MASK]], %[[SCRATCH]], %{{[^,]*}}) : (i32, i32, i32, i32) -> i32
// CHECK: call @__vrc_opencl_dispatch_udononnx_fused_2(%[[SCORES]], %[[SCRATCH]]) : (i32, i32) -> i32
// CHECK: call @__vrc_opencl_dispatch_udononnx_fused_3(%[[SCORES]], %[[SCRATCH]], %{{[^,]*}}) : (i32, i32, i32) -> i32
// CHECK-NOT: call @__vrc_opencl_dispatch_

// The statistics kernel: fragments [0, 864) hold chunk maxima, [864, 1728)
// chunk sums. Chunk c covers [19c, min(19c + 19, 512)); a maximum fragment
// runs no element of the sum loop.
// CHECK-LABEL: gpu.module @udononnx_fused_kernels {
// CHECK: gpu.func @udononnx_fused_0(%[[S:.*]]: memref<16384xf32>, %[[M:.*]]: memref<512xf32>, %{{.*}}: memref<8192xf32>) kernel
// CHECK-SAME: known_grid_size = array<i32: 8192, 1, 1>
// CHECK-SAME: vrc.fragment_cost = {loads = 76 : i64}
// CHECK: %[[LIVE:.*]] = arith.cmpi ult, %[[F:.*]], %c1728_i32 : i32
// CHECK: scf.if %[[LIVE]]
// CHECK: %[[IS_SUM:.*]] = arith.cmpi uge, %[[F]], %c864_i32 : i32
// CHECK: %[[PARTIAL:.*]] = arith.remui %[[F]], %c864_i32 : i32
// CHECK: %[[CHUNK:.*]] = arith.remui %[[PARTIAL]], %c27_i32 : i32
// CHECK: %[[START:.*]] = arith.muli %[[CHUNK]], %c19_i32 : i32
// CHECK: %[[NEXT:.*]] = arith.addi %[[START]], %c19_i32 : i32
// CHECK: %[[END:.*]] = arith.minui %[[NEXT]], %c512_i32 : i32
// CHECK: %[[MAXIMUM:.*]]:2 = scf.while (%[[K:.*]] = %[[START]], %{{.*}} = %{{.*}}) : (i32, f32) -> (i32, f32) {
// CHECK:   arith.cmpi ult, %[[K]], %[[END]] : i32
// CHECK:   memref.load %[[S]][
// CHECK:   memref.load %[[M]][
// CHECK:   arith.addf
// CHECK:   arith.cmpf ogt
// CHECK:   arith.select
// CHECK: %[[SUM_END:.*]] = arith.select %[[IS_SUM]], %[[END]], %[[START]] : i32
// CHECK: %[[SUM:.*]]:2 = scf.while (%[[J:.*]] = %[[START]], %{{.*}} = %{{.*}}) : (i32, f32) -> (i32, f32) {
// CHECK:   arith.cmpi ult, %[[J]], %[[SUM_END]] : i32
// CHECK:   arith.subf %{{.*}}, %[[MAXIMUM]]#1 : f32
// CHECK:   math.exp
// CHECK:   arith.addf
// CHECK: arith.select %[[IS_SUM]], %[[SUM]]#1, %[[MAXIMUM]]#1 : f32

// The elementwise kernel reads the statistics as the scratch surface row.
// The row maximum is the largest chunk maximum, the sum the ascending sum of
// chunk sum * exp(chunk maximum - row maximum).
// CHECK: gpu.func @udononnx_fused_1(%[[S1:.*]]: memref<16384xf32>, %[[M1:.*]]: memref<512xf32>, %[[STATISTICS:.*]]: memref<8192xf32>, %{{.*}}: memref<16384xf32>) kernel
// CHECK-SAME: known_grid_size = array<i32: 8192, 2, 1>
// CHECK-SAME: vrc.fragment_cost = {loads = 83 : i64}
// CHECK: %[[ROW_MAX:.*]]:2 = scf.while
// CHECK:   arith.cmpi ult, %{{.*}}, %c27_i32 : i32
// CHECK:   memref.load %[[STATISTICS]][
// CHECK:   arith.cmpf ogt
// CHECK: %[[ROW_SUM:.*]]:2 = scf.while
// CHECK:   %[[CHUNK_MAX:.*]] = memref.load %[[STATISTICS]][
// CHECK:   %[[SHIFT:.*]] = arith.subf %[[CHUNK_MAX]], %[[ROW_MAX]]#1 : f32
// CHECK:   %[[SCALE:.*]] = math.exp %[[SHIFT]] : f32
// CHECK:   %[[CHUNK_SUM:.*]] = memref.load %[[STATISTICS]][
// CHECK:   %[[SCALED:.*]] = arith.mulf %[[CHUNK_SUM]], %[[SCALE]] : f32
// CHECK:   arith.addf %{{.*}}, %[[SCALED]] : f32
// CHECK: %[[OWN_SCORE:.*]] = memref.load %[[S1]][
// CHECK: %[[OWN_MASK:.*]] = memref.load %[[M1]][
// CHECK: %[[OWN:.*]] = arith.addf %[[OWN_SCORE]], %[[OWN_MASK]] : f32
// CHECK: %[[OWN_SHIFT:.*]] = arith.subf %[[OWN]], %[[ROW_MAX]]#1 : f32
// CHECK: %[[OWN_EXP:.*]] = math.exp %[[OWN_SHIFT]] : f32
// CHECK: arith.divf %[[OWN_EXP]], %[[ROW_SUM]]#1 : f32

// CHECK: gpu.func @udononnx_fused_2(%{{.*}}: memref<16384xf32>, %{{.*}}: memref<8192xf32>) kernel
// CHECK: gpu.func @udononnx_fused_3(%{{.*}}: memref<16384xf32>, %{{.*}}: memref<8192xf32>, %{{.*}}: memref<16384xf32>) kernel
// CHECK-NOT: gpu.func

// PROVEN-COUNT-4: vrc.gather_proven
