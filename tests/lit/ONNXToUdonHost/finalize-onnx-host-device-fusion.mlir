// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph %s | %FileCheck %s --check-prefix=PLAN
// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph --udoncl-materialize-onnx-host-globals --udoncl-materialize-onnx-host-model-registry --udoncl-attach-onnx-host-metadata --udoncl-finalize-onnx-host --udoncl-cleanup-onnx-host-source --canonicalize %s | %FileCheck %s
// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph --udoncl-materialize-onnx-host-globals --udoncl-materialize-onnx-host-model-registry --udoncl-attach-onnx-host-metadata --udoncl-finalize-onnx-host --udoncl-cleanup-onnx-host-source --canonicalize %s | %vrc-opt --vrc-assign-synthesized-kernel-ids=first-id=4096 --vrc-prove-synthesized-gather-kernels | %FileCheck %s --check-prefix=PROVEN
// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph=fuse-device-nodes=false --udoncl-materialize-onnx-host-globals --udoncl-materialize-onnx-host-model-registry --udoncl-attach-onnx-host-metadata --udoncl-finalize-onnx-host --udoncl-cleanup-onnx-host-source --canonicalize %s | %FileCheck %s --check-prefix=UNFUSED

// Trees of per-element and re-indexing Device nodes become one synthesized
// gather kernel each. Nothing here is specific to an operator pattern: the
// rotation of a rotary embedding, a gated product and a broadcast, reshaped,
// transposed and scaled MatMul operand all fall out of the same rule.

func.func private @clEnqueueMarkerWithWaitList(
    !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32

func.func private @__udon_onnx_mul_row(
    tensor<1x4x1x8xf32>, tensor<1x1x1x8xf32>) -> tensor<1x4x1x8xf32>
    attributes {
      udon.onnx.operation = "Mul",
      udon.onnx.provider = "udonatenMul"
    }
func.func private @__udon_onnx_mul_scalar(
    tensor<1x4x1x8xf32>, tensor<f32>) -> tensor<1x4x1x8xf32>
    attributes {
      udon.onnx.operation = "Mul",
      udon.onnx.provider = "udonatenMul"
    }
func.func private @__udon_onnx_mul(
    tensor<1x4x1x8xf32>, tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
    attributes {
      udon.onnx.operation = "Mul",
      udon.onnx.provider = "udonatenMul"
    }
func.func private @__udon_onnx_add(
    tensor<1x4x1x8xf32>, tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
    attributes {
      udon.onnx.operation = "Add",
      udon.onnx.provider = "udonatenAdd"
    }
func.func private @__udon_onnx_slice(
    tensor<1x4x1x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>)
    -> tensor<1x4x1x4xf32>
    attributes {
      udon.onnx.operation = "Slice",
      udon.onnx.provider = "udononnxSlice"
    }
func.func private @__udon_onnx_neg(tensor<1x4x1x4xf32>) -> tensor<1x4x1x4xf32>
    attributes {
      udon.onnx.operation = "Neg",
      udon.onnx.provider = "udononnxNeg"
    }
func.func private @__udon_onnx_concat(
    tensor<1x4x1x4xf32>, tensor<1x4x1x4xf32>) -> tensor<1x4x1x8xf32>
    attributes {
      udon.onnx.operation = "Concat",
      udon.onnx.provider = "udononnxConcat"
    }
func.func private @__udon_onnx_sigmoid(
    tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
    attributes {
      udon.onnx.operation = "Sigmoid",
      udon.onnx.provider = "udonatenSigmoid"
    }
func.func private @__udon_onnx_expand(
    tensor<1x1x2x4xf32>, tensor<4xi64>) -> tensor<1x2x2x4xf32>
    attributes {
      udon.onnx.operation = "Expand",
      udon.onnx.provider = "udononnxExpand"
    }
func.func private @__udon_onnx_reshape(
    tensor<1x2x2x4xf32>, tensor<3xi64>) -> tensor<1x4x4xf32>
    attributes {
      udon.onnx.operation = "Reshape",
      udon.onnx.provider = "udononnxReshape"
    }
func.func private @__udon_onnx_transpose(
    tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
    attributes {
      udon.onnx.operation = "Transpose",
      udon.onnx.provider = "udononnxTranspose"
    }
func.func private @__udon_onnx_mul_keys(
    tensor<1x4x4xf32>, tensor<f32>) -> tensor<1x4x4xf32>
    attributes {
      udon.onnx.operation = "Mul",
      udon.onnx.provider = "udonatenMul"
    }
func.func private @__udon_onnx_matmul(
    tensor<1x4x4xf32>, tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
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
func.func private @__static_zero() -> tensor<1xi64>
    attributes {
      udoncl.static_tensor = {
        byte_length = 8 : i64, id = 1 : i64, data = dense<0> : tensor<1xi64>
      }
    }
func.func private @__static_half() -> tensor<1xi64>
    attributes {
      udoncl.static_tensor = {
        byte_length = 8 : i64, id = 2 : i64, data = dense<4> : tensor<1xi64>
      }
    }
func.func private @__static_end() -> tensor<1xi64>
    attributes {
      udoncl.static_tensor = {
        byte_length = 8 : i64, id = 3 : i64, data = dense<8> : tensor<1xi64>
      }
    }
func.func private @__static_axis() -> tensor<1xi64>
    attributes {
      udoncl.static_tensor = {
        byte_length = 8 : i64, id = 4 : i64, data = dense<3> : tensor<1xi64>
      }
    }
func.func private @__static_expand_shape() -> tensor<4xi64>
    attributes {
      udoncl.static_tensor = {
        byte_length = 32 : i64, id = 5 : i64,
        data = dense<[1, 2, 2, 4]> : tensor<4xi64>
      }
    }
func.func private @__static_keys_shape() -> tensor<3xi64>
    attributes {
      udoncl.static_tensor = {
        byte_length = 24 : i64, id = 6 : i64,
        data = dense<[1, 4, 4]> : tensor<3xi64>
      }
    }

func.func @main_graph(
    %x: tensor<1x4x1x8xf32> {onnx.name = "x"},
    %y: tensor<1x4x1x8xf32> {onnx.name = "y"},
    %cos: tensor<1x1x1x8xf32> {onnx.name = "cos"},
    %sin: tensor<1x1x1x8xf32> {onnx.name = "sin"},
    %keys: tensor<1x1x2x4xf32> {onnx.name = "keys"},
    %queries: tensor<1x4x4xf32> {onnx.name = "queries"})
    -> (tensor<1x4x1x8xf32> {onnx.name = "rotated"},
        tensor<1x4x1x8xf32> {onnx.name = "rotated_again"},
        tensor<1x4x1x8xf32> {onnx.name = "gated"},
        tensor<1x4x4xf32> {onnx.name = "scores"}) {
  %scale = func.call @__static_scale() : () -> tensor<f32>
  %zero = func.call @__static_zero() : () -> tensor<1xi64>
  %half = func.call @__static_half() : () -> tensor<1xi64>
  %end = func.call @__static_end() : () -> tensor<1xi64>
  %axis = func.call @__static_axis() : () -> tensor<1xi64>
  %expand_shape = func.call @__static_expand_shape() : () -> tensor<4xi64>
  %keys_shape = func.call @__static_keys_shape() : () -> tensor<3xi64>

  // (1) Rotation of the two halves of the last axis, scaled by a constant:
  // scale * (x * cos + concat(-x[4:8], x[0:4]) * sin). Eight nodes, one
  // dispatch.
  %x_cos = func.call @__udon_onnx_mul_row(%x, %cos)
      : (tensor<1x4x1x8xf32>, tensor<1x1x1x8xf32>) -> tensor<1x4x1x8xf32>
  %high = func.call @__udon_onnx_slice(%x, %half, %end, %axis)
      : (tensor<1x4x1x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>)
      -> tensor<1x4x1x4xf32>
  %low = func.call @__udon_onnx_slice(%x, %zero, %half, %axis)
      : (tensor<1x4x1x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>)
      -> tensor<1x4x1x4xf32>
  %negated = func.call @__udon_onnx_neg(%high)
      : (tensor<1x4x1x4xf32>) -> tensor<1x4x1x4xf32>
  %turned = func.call @__udon_onnx_concat(%negated, %low)
      {udon.onnx.attributes = {axis = -1 : i64}}
      : (tensor<1x4x1x4xf32>, tensor<1x4x1x4xf32>) -> tensor<1x4x1x8xf32>
  %turned_sin = func.call @__udon_onnx_mul_row(%turned, %sin)
      : (tensor<1x4x1x8xf32>, tensor<1x1x1x8xf32>) -> tensor<1x4x1x8xf32>
  %sum = func.call @__udon_onnx_add(%x_cos, %turned_sin)
      : (tensor<1x4x1x8xf32>, tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %rotated = func.call @__udon_onnx_mul_scalar(%sum, %scale)
      : (tensor<1x4x1x8xf32>, tensor<f32>) -> tensor<1x4x1x8xf32>

  // (5) The same expression over other buffers shares the kernel of (1).
  %y_cos = func.call @__udon_onnx_mul_row(%y, %cos)
      : (tensor<1x4x1x8xf32>, tensor<1x1x1x8xf32>) -> tensor<1x4x1x8xf32>
  %y_high = func.call @__udon_onnx_slice(%y, %half, %end, %axis)
      : (tensor<1x4x1x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>)
      -> tensor<1x4x1x4xf32>
  %y_low = func.call @__udon_onnx_slice(%y, %zero, %half, %axis)
      : (tensor<1x4x1x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>)
      -> tensor<1x4x1x4xf32>
  %y_negated = func.call @__udon_onnx_neg(%y_high)
      : (tensor<1x4x1x4xf32>) -> tensor<1x4x1x4xf32>
  %y_turned = func.call @__udon_onnx_concat(%y_negated, %y_low)
      {udon.onnx.attributes = {axis = -1 : i64}}
      : (tensor<1x4x1x4xf32>, tensor<1x4x1x4xf32>) -> tensor<1x4x1x8xf32>
  %y_turned_sin = func.call @__udon_onnx_mul_row(%y_turned, %sin)
      : (tensor<1x4x1x8xf32>, tensor<1x1x1x8xf32>) -> tensor<1x4x1x8xf32>
  %y_sum = func.call @__udon_onnx_add(%y_cos, %y_turned_sin)
      : (tensor<1x4x1x8xf32>, tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %rotated_again = func.call @__udon_onnx_mul_scalar(%y_sum, %scale)
      : (tensor<1x4x1x8xf32>, tensor<f32>) -> tensor<1x4x1x8xf32>

  // (2) A gated product: (x * sigmoid(x)) * y. Three nodes, one dispatch.
  %gate = func.call @__udon_onnx_sigmoid(%x)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %activated = func.call @__udon_onnx_mul(%x, %gate)
      : (tensor<1x4x1x8xf32>, tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %gated = func.call @__udon_onnx_mul(%activated, %y)
      : (tensor<1x4x1x8xf32>, tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>

  // (3) Expand, Reshape, Transpose and a scalar Mul feed a MatMul operand:
  // one dispatch for the operand and the SGEMM.
  %expanded = func.call @__udon_onnx_expand(%keys, %expand_shape)
      : (tensor<1x1x2x4xf32>, tensor<4xi64>) -> tensor<1x2x2x4xf32>
  %merged = func.call @__udon_onnx_reshape(%expanded, %keys_shape)
      : (tensor<1x2x2x4xf32>, tensor<3xi64>) -> tensor<1x4x4xf32>
  %transposed = func.call @__udon_onnx_transpose(%merged)
      {udon.onnx.attributes = {perm = [0, 2, 1]}}
      : (tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
  %scaled = func.call @__udon_onnx_mul_keys(%transposed, %scale)
      : (tensor<1x4x4xf32>, tensor<f32>) -> tensor<1x4x4xf32>
  %scores = func.call @__udon_onnx_matmul(%queries, %scaled)
      : (tensor<1x4x4xf32>, tensor<1x4x4xf32>) -> tensor<1x4x4xf32>

  return %rotated, %rotated_again, %gated, %scores
      : tensor<1x4x1x8xf32>, tensor<1x4x1x8xf32>, tensor<1x4x1x8xf32>,
        tensor<1x4x4xf32>
}

// The plan names the root of every node: the eight nodes of each rotation,
// the three of the gated product, the four of the MatMul operand, and the
// MatMul on its own. Only the scaled operand, which the MatMul reads, owns a
// Device buffer.
// PLAN: udoncl.onnx_host_plan = {
// PLAN-SAME: device_fusion_roots = array<i64: 7, 7, 7, 7, 7, 7, 7, 7, 15, 15, 15, 15, 15, 15, 15, 15, 18, 18, 18, 22, 22, 22, 22, 23>
// PLAN-SAME: owned_device_buffers = array<i64: 35>

// CHECK-LABEL: func.func @udononnxGeneratedRun(
// CHECK-NOT: call @__vrc_opencl_dispatch_{{(udondnn|udonblas|udononnx_[^f])}}
// CHECK: call @__vrc_opencl_dispatch_udononnx_fused_0(%[[X:[^,]*]], %[[COS:[^,]*]], %[[SIN:[^,]*]], %{{[^,]*}}) : (i32, i32, i32, i32) -> i32
// CHECK-NEXT: call @__vrc_opencl_await(
// CHECK-NOT: call @__vrc_opencl_dispatch_{{(udondnn|udonblas|udononnx_[^f])}}
// CHECK: call @__vrc_opencl_dispatch_udononnx_fused_0(%[[Y:[^,]*]], %[[COS]], %[[SIN]], %{{[^,]*}}) : (i32, i32, i32, i32) -> i32
// CHECK-NOT: call @__vrc_opencl_dispatch_{{(udondnn|udonblas|udononnx_[^f])}}
// CHECK: call @__vrc_opencl_dispatch_udononnx_fused_1(%[[X]], %[[Y]], %{{[^,]*}}) : (i32, i32, i32) -> i32
// CHECK-NOT: call @__vrc_opencl_dispatch_{{(udondnn|udonblas|udononnx_[^f])}}
// CHECK: call @__vrc_opencl_dispatch_udononnx_fused_2(%[[KEYS:[^,]*]], %[[SCALED:[^,]*]]) : (i32, i32) -> i32
// CHECK-NOT: call @__vrc_opencl_dispatch_{{(udondnn|udononnx)}}
// CHECK: call @__vrc_opencl_dispatch_udonblas_arena_sgemm(%{{[^,]*}}, %[[SCALED]],
// CHECK-NOT: call @__vrc_opencl_dispatch_

// CHECK-LABEL: gpu.module @udononnx_fused_kernels {

// (1) scale * (x * cos + concat(-x[4:8], x[0:4]) * sin) over 32 elements.
// A fragment reads x, cos, one side of the Concat and sin.
// CHECK: gpu.func @udononnx_fused_0(%[[X:.*]]: memref<32xf32>, %[[COS:.*]]: memref<8xf32>, %[[SIN:.*]]: memref<8xf32>, %[[OUT:.*]]: memref<8192xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 8192, 1, 1>, vrc.fragment_cost = {loads = 4 : i64}, vrc.profile = "gather-f32", vrc.synthesized_kernel, vrc.words = 8192 : i32} {
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[SCALE:.*]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG: %[[C3:.*]] = arith.constant 3 : i32
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : i32
// CHECK-DAG: %[[C8:.*]] = arith.constant 8 : i32
// CHECK-DAG: %[[C32:.*]] = arith.constant 32 : i32
// CHECK-DAG: %[[WIDTH:.*]] = arith.constant 8192 : index
// CHECK: %[[GX:.*]] = gpu.global_id x
// CHECK: %[[GY:.*]] = gpu.global_id y
// CHECK: %[[ROW:.*]] = arith.muli %[[GY]], %[[WIDTH]] : index
// CHECK: %[[FRAGMENT:.*]] = arith.addi %[[GX]], %[[ROW]] : index
// CHECK: %[[I:.*]] = arith.index_cast %[[FRAGMENT]] : index to i32
// CHECK: %[[LIVE:.*]] = arith.cmpi ult, %[[I]], %[[C32]] : i32
// CHECK: %[[VALUE:.*]] = scf.if %[[LIVE]] -> (f32) {
// CHECK:   %[[I_INDEX:.*]] = arith.index_cast %[[I]] : i32 to index
// CHECK:   %[[X_VALUE:.*]] = memref.load %[[X]][%[[I_INDEX]]] : memref<32xf32>
// CHECK:   %[[LINE:.*]] = arith.divui %[[I]], %[[C8]] : i32
// CHECK:   %[[COLUMN:.*]] = arith.remui %[[I]], %[[C8]] : i32
// CHECK:   %[[COLUMN_INDEX:.*]] = arith.index_cast %[[COLUMN]] : i32 to index
// CHECK:   %[[COS_VALUE:.*]] = memref.load %[[COS]][%[[COLUMN_INDEX]]] : memref<8xf32>
// CHECK:   %[[X_COS:.*]] = arith.mulf %[[X_VALUE]], %[[COS_VALUE]] : f32
// CHECK:   %[[IS_LEFT:.*]] = arith.cmpi ult, %[[COLUMN]], %[[C4]] : i32
// CHECK:   %[[TURNED:.*]] = scf.if %[[IS_LEFT]] -> (f32) {
// The index bounds do not depend on the branch: a minimum bounds the
// coordinate of each side.
// CHECK:     %[[LEFT:.*]] = arith.minui %[[COLUMN]], %[[C3]] : i32
// CHECK:     %[[HIGH:.*]] = arith.addi %[[LEFT]], %[[C4]] : i32
// CHECK:     %[[LINE_BASE:.*]] = arith.muli %[[LINE]], %[[C8]] : i32
// CHECK:     %[[HIGH_LINEAR:.*]] = arith.addi %[[LINE_BASE]], %[[HIGH]] : i32
// CHECK:     %[[HIGH_INDEX:.*]] = arith.index_cast %[[HIGH_LINEAR]] : i32 to index
// CHECK:     %[[HIGH_VALUE:.*]] = memref.load %[[X]][%[[HIGH_INDEX]]] : memref<32xf32>
// CHECK:     %[[NEGATED:.*]] = arith.negf %[[HIGH_VALUE]] : f32
// CHECK:     scf.yield %[[NEGATED]] : f32
// CHECK:   } else {
// CHECK:     %[[MOVED:.*]] = arith.subi %[[COLUMN]], %[[C4]] : i32
// CHECK:     %[[RIGHT:.*]] = arith.minui %[[MOVED]], %[[C3]] : i32
// CHECK:     %[[LOW_BASE:.*]] = arith.muli %[[LINE]], %[[C8]] : i32
// CHECK:     %[[LOW_LINEAR:.*]] = arith.addi %[[LOW_BASE]], %[[RIGHT]] : i32
// CHECK:     %[[LOW_INDEX:.*]] = arith.index_cast %[[LOW_LINEAR]] : i32 to index
// CHECK:     %[[LOW_VALUE:.*]] = memref.load %[[X]][%[[LOW_INDEX]]] : memref<32xf32>
// CHECK:     scf.yield %[[LOW_VALUE]] : f32
// CHECK:   }
// CHECK:   %[[SIN_VALUE:.*]] = memref.load %[[SIN]][%{{.*}}] : memref<8xf32>
// CHECK:   %[[TURNED_SIN:.*]] = arith.mulf %[[TURNED]], %[[SIN_VALUE]] : f32
// CHECK:   %[[SUM:.*]] = arith.addf %[[X_COS]], %[[TURNED_SIN]] : f32
// CHECK:   %[[SCALED:.*]] = arith.mulf %[[SUM]], %[[SCALE]] : f32
// CHECK:   scf.yield %[[SCALED]] : f32
// CHECK: } else {
// CHECK:   scf.yield %[[ZERO]] : f32
// CHECK: }
// CHECK: memref.store %[[VALUE]], %[[OUT]][%[[FRAGMENT]]] : memref<8192xf32>
// CHECK: gpu.return

// (2) (x * sigmoid(x)) * y. Both readers of x share one Load, and the
// sigmoid is the library expression 1 / (1 + exp(-x)) with a NaN passed on.
// CHECK: gpu.func @udononnx_fused_1(%[[X:.*]]: memref<32xf32>, %[[Y:.*]]: memref<32xf32>, %{{.*}}: memref<8192xf32>) kernel
// CHECK-SAME: vrc.fragment_cost = {loads = 2 : i64}
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[X_VALUE:.*]] = memref.load %[[X]][
// CHECK-NOT: memref.load %[[X]][
// CHECK: %[[MINUS:.*]] = arith.negf %[[X_VALUE]] : f32
// CHECK: %[[EXP:.*]] = math.exp %[[MINUS]] : f32
// CHECK: %[[DENOMINATOR:.*]] = arith.addf %[[EXP]], %[[ONE]] : f32
// CHECK: %[[SIGMOID:.*]] = arith.divf %[[ONE]], %[[DENOMINATOR]] : f32
// CHECK: %[[NAN:.*]] = arith.cmpf une, %[[X_VALUE]], %[[X_VALUE]] : f32
// CHECK: %[[GATE:.*]] = arith.select %[[NAN]], %[[X_VALUE]], %[[SIGMOID]] : f32
// CHECK: %[[ACTIVATED:.*]] = arith.mulf %[[X_VALUE]], %[[GATE]] : f32
// CHECK: %[[Y_VALUE:.*]] = memref.load %[[Y]][
// CHECK: %[[GATED:.*]] = arith.mulf %[[ACTIVATED]], %[[Y_VALUE]] : f32
// CHECK: scf.yield %[[GATED]] : f32

// (3) 0.5 * transpose(reshape(expand(keys))): pure index arithmetic in front
// of one Load. Element (i, j) of the 4 x 4 result reads row j % 2, column i.
// CHECK: gpu.func @udononnx_fused_2(%[[KEYS:.*]]: memref<8xf32>, %{{.*}}: memref<8192xf32>) kernel
// CHECK-SAME: vrc.fragment_cost = {loads = 1 : i64}
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : i32
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : i32
// CHECK: %[[I:.*]] = arith.index_cast %{{.*}} : index to i32
// CHECK: %[[OUTER:.*]] = arith.divui %[[I]], %[[C4]] : i32
// CHECK: %[[INNER:.*]] = arith.remui %[[I]], %[[C4]] : i32
// CHECK: %[[SWAPPED_BASE:.*]] = arith.muli %[[INNER]], %[[C4]] : i32
// CHECK: %[[SWAPPED:.*]] = arith.addi %[[SWAPPED_BASE]], %[[OUTER]] : i32
// CHECK: %[[MERGED_ROW:.*]] = arith.divui %[[SWAPPED]], %[[C4]] : i32
// CHECK: %[[SOURCE_ROW:.*]] = arith.remui %[[MERGED_ROW]], %[[C2]] : i32
// CHECK: %[[SOURCE_COLUMN:.*]] = arith.remui %[[SWAPPED]], %[[C4]] : i32
// CHECK: %[[SOURCE_BASE:.*]] = arith.muli %[[SOURCE_ROW]], %[[C4]] : i32
// CHECK: %[[SOURCE:.*]] = arith.addi %[[SOURCE_BASE]], %[[SOURCE_COLUMN]] : i32
// CHECK: %[[SOURCE_INDEX:.*]] = arith.index_cast %[[SOURCE]] : i32 to index
// CHECK: %[[KEY:.*]] = memref.load %[[KEYS]][%[[SOURCE_INDEX]]] : memref<8xf32>
// CHECK: arith.mulf %[[KEY]], %{{.*}} : f32

// (5) Two rotations, three kernels: no fourth kernel and one declaration
// per kernel.
// CHECK-NOT: gpu.func
// CHECK: func.func private @__vrc_opencl_dispatch_udononnx_fused_0(i32, i32, i32, i32) -> i32 attributes {vrc.synthesized_dispatch = @udononnx_fused_kernels::@udononnx_fused_0}
// CHECK: func.func private @__vrc_opencl_dispatch_udononnx_fused_1(i32, i32, i32) -> i32 attributes {vrc.synthesized_dispatch = @udononnx_fused_kernels::@udononnx_fused_1}
// CHECK: func.func private @__vrc_opencl_dispatch_udononnx_fused_2(i32, i32) -> i32 attributes {vrc.synthesized_dispatch = @udononnx_fused_kernels::@udononnx_fused_2}

// The kernels are still provable after cleanup and canonicalization, and
// their declarations describe the dispatch without a name table.
// PROVEN: gpu.module @device_gpu_4096 {
// PROVEN-NEXT: gpu.func @udononnx_fused_0({{.*}}vrc.gather_proven, vrc.kernel_id = 4096 : i32
// PROVEN: gpu.module @device_gpu_4097 {
// PROVEN-NEXT: gpu.func @udononnx_fused_1({{.*}}vrc.gather_proven, vrc.kernel_id = 4097 : i32
// PROVEN: gpu.module @device_gpu_4098 {
// PROVEN-NEXT: gpu.func @udononnx_fused_2({{.*}}vrc.gather_proven, vrc.kernel_id = 4098 : i32
// PROVEN-NOT: gpu.module
// PROVEN: func.func private @__vrc_opencl_dispatch_udononnx_fused_0(i32, i32, i32, i32) -> i32 attributes {vrc.device_dispatch = {buffer_count = 4 : i32, fragment_cost = {loads = 4 : i64}, global_size = array<i32: 8192, 1, 1>, kernel_id = 4096 : i32}}
// PROVEN: func.func private @__vrc_opencl_dispatch_udononnx_fused_1(i32, i32, i32) -> i32 attributes {vrc.device_dispatch = {buffer_count = 3 : i32, fragment_cost = {loads = 2 : i64}, global_size = array<i32: 8192, 1, 1>, kernel_id = 4097 : i32}}
// PROVEN: func.func private @__vrc_opencl_dispatch_udononnx_fused_2(i32, i32) -> i32 attributes {vrc.device_dispatch = {buffer_count = 2 : i32, fragment_cost = {loads = 1 : i64}, global_size = array<i32: 8192, 1, 1>, kernel_id = 4098 : i32}}

// With fusion off every node dispatches its library kernel as before and the
// module carries no kernel.
// UNFUSED-LABEL: func.func @udononnxGeneratedRun(
// UNFUSED: call @__vrc_opencl_dispatch_udondnn_large_op_tensor
// UNFUSED: call @__vrc_opencl_dispatch_udononnx_slice_f32
// UNFUSED: call @__vrc_opencl_dispatch_udononnx_slice_f32
// UNFUSED: call @__vrc_opencl_dispatch_udononnx_unary_f32
// UNFUSED: call @__vrc_opencl_dispatch_udononnx_concat2_f32
// UNFUSED: call @__vrc_opencl_dispatch_udondnn_large_op_tensor
// UNFUSED: call @__vrc_opencl_dispatch_udondnn_large_op_tensor
// UNFUSED: call @__vrc_opencl_dispatch_udondnn_large_op_tensor
// UNFUSED: call @__vrc_opencl_dispatch_udondnn_large_op_tensor
// UNFUSED: call @__vrc_opencl_dispatch_udononnx_slice_f32
// UNFUSED: call @__vrc_opencl_dispatch_udononnx_slice_f32
// UNFUSED: call @__vrc_opencl_dispatch_udononnx_unary_f32
// UNFUSED: call @__vrc_opencl_dispatch_udononnx_concat2_f32
// UNFUSED: call @__vrc_opencl_dispatch_udondnn_large_op_tensor
// UNFUSED: call @__vrc_opencl_dispatch_udondnn_large_op_tensor
// UNFUSED: call @__vrc_opencl_dispatch_udondnn_large_op_tensor
// UNFUSED: call @__vrc_opencl_dispatch_udondnn_large_activation
// UNFUSED: call @__vrc_opencl_dispatch_udondnn_large_op_tensor
// UNFUSED: call @__vrc_opencl_dispatch_udondnn_large_op_tensor
// UNFUSED: call @__vrc_opencl_dispatch_udononnx_expand_f32
// UNFUSED: call @__vrc_opencl_dispatch_udononnx_transpose_f32
// UNFUSED: call @__vrc_opencl_dispatch_udondnn_large_op_tensor
// UNFUSED: call @__vrc_opencl_dispatch_udonblas_arena_sgemm
// UNFUSED-NOT: call @__vrc_opencl_dispatch_
// UNFUSED-NOT: gpu.module
// UNFUSED-NOT: vrc.synthesized_dispatch
