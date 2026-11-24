// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph --udoncl-materialize-onnx-host-globals --udoncl-materialize-onnx-host-model-registry --udoncl-attach-onnx-host-metadata --udoncl-finalize-onnx-host --udoncl-cleanup-onnx-host-source --canonicalize %s | %FileCheck %s

// A graph with the bounded dimension `past`: the input past has extent past,
// the input mask and the output viewed have extent past + 1. The registry
// lists -1 dimensions with their largest extents, BindMemory keeps the extent
// and the pitch of every dynamic binding (a pitch beyond the largest extent
// of `past` on any binding, 8, is not supported), and the run derives `past`
// from the first binding and checks the others against it.

module attributes {udoncl.dimension_bounds = {past = 7 : i64}} {
  func.func private @clEnqueueMarkerWithWaitList(
      !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
  func.func private @__udon_onnx_unsqueeze(tensor<1x?xf32>, tensor<1xi64>)
      -> tensor<1x1x?xf32>
      attributes {udon.onnx.operation = "Unsqueeze",
                  udon.onnx.provider = "udononnxUnsqueeze"}
  func.func private @__udoncl_static_tensor_0() -> tensor<1xi64>
      attributes {udoncl.static_tensor = {alignment = 8 : i64, byte_length = 8 : i64, byte_strides = array<i64: 8>, data = dense<0> : tensor<1xi64>, data_type = "int64", id = 0 : i32, layout = "row_major_contiguous", name = "constant_0", sha256 = "", shape = array<i64: 1>, symbol = "__udoncl_static_tensor_0"}}

  func.func @main_graph(
      %past: tensor<1x?xf32> {onnx.name = "past", udoncl.dim_exprs = [{axis = 1 : i64, offset = 0 : i64, symbol = "past"}]},
      %mask: tensor<1x?xf32> {onnx.name = "mask", udoncl.dim_exprs = [{axis = 1 : i64, offset = 1 : i64, symbol = "past"}]})
      -> (tensor<1x1x?xf32> {onnx.name = "viewed"}) {
    %axis = func.call @__udoncl_static_tensor_0() : () -> tensor<1xi64>
    %viewed = func.call @__udon_onnx_unsqueeze(%mask, %axis)
        {udon.onnx.result_dim_expr = {axis = 2 : i64, offset = 1 : i64, symbol = "past"}}
        : (tensor<1x?xf32>, tensor<1xi64>) -> tensor<1x1x?xf32>
    return %viewed : tensor<1x1x?xf32>
  }
}

// CHECK-DAG: "udonhl.global"() <{{{.*}}sym_name = "__udon_onnx_dimension_0"
// CHECK-DAG: "udonhl.global"() <{{{.*}}sym_name = "__udon_onnx_bound_input_extent_0"
// CHECK-DAG: "udonhl.global"() <{{{.*}}sym_name = "__udon_onnx_bound_input_pitch_1"
// CHECK-DAG: "udonhl.global"() <{{{.*}}sym_name = "__udon_onnx_bound_output_extent_0"
// CHECK-DAG: "udonhl.global"() <{{{.*}}sym_name = "__udon_onnx_input_dimension_bounds_1"

// CHECK-LABEL: func.func @udononnxGeneratedBindMemory(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: !udonhl.ptr<0>, %arg4: i32, %arg5: i32) -> i32
// CHECK: arith.cmpi ule, %arg5, %c8_i32
// CHECK: udonhl.address_of @__udon_onnx_bound_input_extent_0
// CHECK: udonhl.address_of @__udon_onnx_bound_input_pitch_0

// CHECK-LABEL: func.func @udononnxGeneratedRun(
// CHECK: udonhl.address_of @__udon_onnx_bound_input_extent_0
// CHECK-NEXT: %[[PAST:.*]] = "udonhl.load"
// CHECK-NEXT: arith.cmpi sge, %[[PAST]], %c0_i32
// CHECK: udonhl.address_of @__udon_onnx_bound_input_extent_1
// CHECK: arith.addi %[[PAST]], %c1_i32
// CHECK: udonhl.address_of @__udon_onnx_bound_output_extent_0
// CHECK: arith.addi %[[PAST]], %c1_i32
// CHECK: udonhl.address_of @__udon_onnx_dimension_0
// CHECK-NEXT: "udonhl.store"(%[[PAST]],
