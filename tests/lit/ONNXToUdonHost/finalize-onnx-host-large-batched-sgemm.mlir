// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph --udoncl-materialize-onnx-host-globals --udoncl-materialize-onnx-host-model-registry --udoncl-attach-onnx-host-metadata --udoncl-finalize-onnx-host --udoncl-cleanup-onnx-host-source --canonicalize %s | %FileCheck %s

// Batched MatMuls beyond udonblas_arena_sgemm_strided_batched (operands of
// 131072 words, outputs of 16384) run its large variant, whose operands may be
// whole Device buffer textures and whose surface holds 131072 words: the
// scores of 32 queries against 4096 keys and their weighted sum of 4096
// values. A product within the small kernel keeps it.

func.func private @clEnqueueMarkerWithWaitList(
    !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32

func.func private @__udon_onnx_matmul_scores(
    tensor<1x4x8x64xf32>, tensor<1x4x64x4096xf32>) -> tensor<1x4x8x4096xf32>
    attributes {
      udon.onnx.operation = "MatMul",
      udon.onnx.provider = "udonblasSgemm"
    }
func.func private @__udon_onnx_matmul_values(
    tensor<1x4x8x4096xf32>, tensor<1x4x4096x64xf32>) -> tensor<1x4x8x64xf32>
    attributes {
      udon.onnx.operation = "MatMul",
      udon.onnx.provider = "udonblasSgemm"
    }
func.func private @__udon_onnx_matmul_small(
    tensor<1x4x8x64xf32>, tensor<1x4x64x512xf32>) -> tensor<1x4x8x512xf32>
    attributes {
      udon.onnx.operation = "MatMul",
      udon.onnx.provider = "udonblasSgemm"
    }

func.func @main_graph(
    %queries: tensor<1x4x8x64xf32> {onnx.name = "queries"},
    %keys: tensor<1x4x64x4096xf32> {onnx.name = "keys"},
    %weights: tensor<1x4x8x4096xf32> {onnx.name = "weights"},
    %values: tensor<1x4x4096x64xf32> {onnx.name = "values"},
    %short_keys: tensor<1x4x64x512xf32> {onnx.name = "short_keys"})
    -> (tensor<1x4x8x4096xf32> {onnx.name = "scores"},
        tensor<1x4x8x64xf32> {onnx.name = "attended"},
        tensor<1x4x8x512xf32> {onnx.name = "short_scores"}) {
  %scores = func.call @__udon_onnx_matmul_scores(%queries, %keys)
      : (tensor<1x4x8x64xf32>, tensor<1x4x64x4096xf32>)
      -> tensor<1x4x8x4096xf32>
  %attended = func.call @__udon_onnx_matmul_values(%weights, %values)
      : (tensor<1x4x8x4096xf32>, tensor<1x4x4096x64xf32>)
      -> tensor<1x4x8x64xf32>
  %short = func.call @__udon_onnx_matmul_small(%queries, %short_keys)
      : (tensor<1x4x8x64xf32>, tensor<1x4x64x512xf32>)
      -> tensor<1x4x8x512xf32>
  return %scores, %attended, %short
      : tensor<1x4x8x4096xf32>, tensor<1x4x8x64xf32>, tensor<1x4x8x512xf32>
}

// m, n, k, the strides of the three operands and the batch count.
// CHECK-LABEL: func.func @udononnxGeneratedEnqueueRun
// CHECK-DAG: %[[C8:.*]] = arith.constant 8 : i32
// CHECK-DAG: %[[C64:.*]] = arith.constant 64 : i32
// CHECK-DAG: %[[C4096:.*]] = arith.constant 4096 : i32
// CHECK-DAG: %[[C512:.*]] = arith.constant 512 : i32
// CHECK-DAG: %[[C32768:.*]] = arith.constant 32768 : i32
// CHECK-DAG: %[[C262144:.*]] = arith.constant 262144 : i32
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : i32
// CHECK: call @__vrc_opencl_dispatch_udonblas_arena_sgemm_strided_batched_large(%{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %[[C8]], %[[C4096]], %[[C64]], %[[C512]], %[[C262144]], %[[C32768]], %[[C4]],
// CHECK: call @__vrc_opencl_dispatch_udonblas_arena_sgemm_strided_batched_large(%{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %[[C8]], %[[C64]], %[[C4096]], %[[C32768]], %[[C262144]], %[[C512]], %[[C4]],
// CHECK: call @__vrc_opencl_dispatch_udonblas_arena_sgemm_strided_batched(%{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %[[C8]], %[[C512]], %[[C64]], %[[C512]], %[[C32768]], %[[C4096]], %[[C4]],
