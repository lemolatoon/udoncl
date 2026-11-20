// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph --udoncl-materialize-onnx-host-globals --udoncl-materialize-onnx-host-model-registry --udoncl-attach-onnx-host-metadata --udoncl-finalize-onnx-host --udoncl-cleanup-onnx-host-source --canonicalize %s | %FileCheck %s
// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph --udoncl-materialize-onnx-host-globals --udoncl-materialize-onnx-host-model-registry --udoncl-attach-onnx-host-metadata --udoncl-finalize-onnx-host --udoncl-cleanup-onnx-host-source --canonicalize %s | %vrc-opt --vrc-assign-synthesized-kernel-ids=first-id=4096 --vrc-prove-synthesized-gather-kernels | %FileCheck %s --check-prefix=PROVEN

// The bounds proof of a fused kernel must survive the canonicalization that
// follows Host finalization. A Slice of a Concat is the hard case: the Slice
// offsets the Concat coordinate, and canonicalization folds that offset into
// the subtraction on the right side of the Concat, leaving an addition of a
// negative constant that neither interval arithmetic nor the branch
// condition bounds. The minimum the synthesis puts on each side does.

func.func private @clEnqueueMarkerWithWaitList(
    !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32

func.func private @__udon_onnx_concat(
    tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x16xf32>
    attributes {
      udon.onnx.operation = "Concat",
      udon.onnx.provider = "udononnxConcat"
    }
func.func private @__udon_onnx_slice(
    tensor<4x16xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>)
    -> tensor<4x12xf32>
    attributes {
      udon.onnx.operation = "Slice",
      udon.onnx.provider = "udononnxSlice"
    }
func.func private @__static_start() -> tensor<1xi64>
    attributes {
      udoncl.static_tensor = {
        byte_length = 8 : i64, id = 0 : i64, data = dense<2> : tensor<1xi64>
      }
    }
func.func private @__static_end() -> tensor<1xi64>
    attributes {
      udoncl.static_tensor = {
        byte_length = 8 : i64, id = 1 : i64, data = dense<14> : tensor<1xi64>
      }
    }
func.func private @__static_axis() -> tensor<1xi64>
    attributes {
      udoncl.static_tensor = {
        byte_length = 8 : i64, id = 2 : i64, data = dense<1> : tensor<1xi64>
      }
    }

func.func @main_graph(
    %x: tensor<4x8xf32> {onnx.name = "x"},
    %y: tensor<4x8xf32> {onnx.name = "y"})
    -> (tensor<4x12xf32> {onnx.name = "middle"}) {
  %start = func.call @__static_start() : () -> tensor<1xi64>
  %end = func.call @__static_end() : () -> tensor<1xi64>
  %axis = func.call @__static_axis() : () -> tensor<1xi64>
  %joined = func.call @__udon_onnx_concat(%x, %y)
      {udon.onnx.attributes = {axis = 1 : i64}}
      : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x16xf32>
  %middle = func.call @__udon_onnx_slice(%joined, %start, %end, %axis)
      : (tensor<4x16xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>)
      -> tensor<4x12xf32>
  return %middle : tensor<4x12xf32>
}

// CHECK: call @__vrc_opencl_dispatch_udononnx_fused_0(
// CHECK-NOT: call @__vrc_opencl_dispatch_

// CHECK: gpu.func @udononnx_fused_0(%[[X:.*]]: memref<32xf32>, %[[Y:.*]]: memref<32xf32>, %{{.*}}: memref<8192xf32>) kernel
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : i32
// CHECK-DAG: %[[C7:.*]] = arith.constant 7 : i32
// CHECK-DAG: %[[C8:.*]] = arith.constant 8 : i32
// CHECK-DAG: %[[C12:.*]] = arith.constant 12 : i32
// CHECK-DAG: %[[MINUS6:.*]] = arith.constant -6 : i32
// CHECK: %[[LINE:.*]] = arith.divui %[[I:.*]], %[[C12]] : i32
// CHECK: %[[COLUMN:.*]] = arith.remui %[[I]], %[[C12]] : i32
// CHECK: %[[JOINED:.*]] = arith.addi %[[COLUMN]], %[[C2]] : i32
// CHECK: %[[IS_LEFT:.*]] = arith.cmpi ult, %[[JOINED]], %[[C8]] : i32
// CHECK: scf.if %[[IS_LEFT]] -> (f32) {
// CHECK:   %[[LEFT:.*]] = arith.minui %[[JOINED]], %[[C7]] : i32
// CHECK:   memref.load %[[X]][
// CHECK: } else {
// CHECK:   %[[MOVED:.*]] = arith.addi %[[COLUMN]], %[[MINUS6]] : i32
// CHECK:   %[[RIGHT:.*]] = arith.minui %[[MOVED]], %[[C7]] : i32
// CHECK:   memref.load %[[Y]][
// CHECK: }

// PROVEN: gpu.func @udononnx_fused_0({{.*}}vrc.gather_proven
