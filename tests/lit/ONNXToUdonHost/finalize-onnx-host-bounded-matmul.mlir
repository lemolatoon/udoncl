// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph --udoncl-materialize-onnx-host-globals --udoncl-materialize-onnx-host-model-registry --udoncl-attach-onnx-host-metadata --udoncl-finalize-onnx-host --udoncl-cleanup-onnx-host-source --canonicalize %s | %FileCheck %s

// MatMul over the bounded dimension `past`. The first product has past rows
// (m) and writes them dense; the second contracts over them (k). Both run
// the strided-batched SGEMM the largest extent needs, with m and k the
// run-time value of past (loaded once per run).

module attributes {udoncl.dimension_bounds = {past = 7 : i64}} {
  func.func private @clEnqueueMarkerWithWaitList(
      !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
  func.func private @__udon_onnx_matmul_rows(tensor<1x?x4xf32>, tensor<4x3xf32>)
      -> tensor<1x?x3xf32>
      attributes {udon.onnx.operation = "MatMul",
                  udon.onnx.provider = "udonblasSgemm"}
  func.func private @__udon_onnx_matmul_inner(tensor<1x1x?xf32>, tensor<1x?x3xf32>)
      -> tensor<1x1x3xf32>
      attributes {udon.onnx.operation = "MatMul",
                  udon.onnx.provider = "udonblasSgemm"}

  func.func @main_graph(
      %rows: tensor<1x?x4xf32> {onnx.name = "rows", udoncl.dim_exprs = [{axis = 1 : i64, offset = 0 : i64, symbol = "past"}]},
      %weights: tensor<4x3xf32> {onnx.name = "weights"},
      %probabilities: tensor<1x1x?xf32> {onnx.name = "probabilities", udoncl.dim_exprs = [{axis = 2 : i64, offset = 0 : i64, symbol = "past"}]})
      -> (tensor<1x1x3xf32> {onnx.name = "attention"}) {
    %values = func.call @__udon_onnx_matmul_rows(%rows, %weights)
        {udon.onnx.result_dim_expr = {axis = 1 : i64, offset = 0 : i64, symbol = "past"}}
        : (tensor<1x?x4xf32>, tensor<4x3xf32>) -> tensor<1x?x3xf32>
    %attention = func.call @__udon_onnx_matmul_inner(%probabilities, %values)
        : (tensor<1x1x?xf32>, tensor<1x?x3xf32>) -> tensor<1x1x3xf32>
    return %attention : tensor<1x1x3xf32>
  }
}

// CHECK-LABEL: func.func private @__udononnx_enqueue_resolved_run(%arg0
// CHECK: udonhl.address_of @__udon_onnx_dimension_0
// CHECK-NEXT: %[[PAST:.*]] = "udonhl.load"
// CHECK: %[[WORDS:.*]] = arith.muli %[[PAST]], %c3_i32
// CHECK: call @__vrc_opencl_dispatch_udonblas_arena_sgemm_strided_batched(%{{.*}}, %{{.*}}, %{{.*}}, %[[PAST]], %c3_i32, %c4_i32, %c0_i32, %c0_i32, %[[WORDS]], %c1_i32, %c0_i32, %c0_i32, %cst) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, f32) -> i32
// CHECK: call @__vrc_opencl_dispatch_udonblas_arena_sgemm_strided_batched(%{{.*}}, %{{.*}}, %{{.*}}, %c1_i32, %c3_i32, %[[PAST]], %c0_i32, %c0_i32, %c3_i32, %c1_i32, %c0_i32, %c0_i32, %cst) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, f32) -> i32
