// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph --udoncl-materialize-onnx-host-globals --udoncl-materialize-onnx-host-model-registry --udoncl-attach-onnx-host-metadata --udoncl-finalize-onnx-host --udoncl-cleanup-onnx-host-source --canonicalize %s | %FileCheck %s

// The padding of an attention mask of past + 1 positions: Cast, Equal and
// Where run their library kernels with the run-time element count past + 1
// and that extent packed into the high half of their shape words. The
// bounded prefix reads the leading block of its constant at the bound; the
// mask input and the output have one row, so their pitch does not matter.

module attributes {udoncl.dimension_bounds = {past = 7 : i64}} {
  func.func private @clEnqueueMarkerWithWaitList(
      !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
  func.func private @__udon_onnx_cast(tensor<1x?xi64>) -> tensor<1x?xf32>
      attributes {udon.onnx.operation = "Cast", udon.onnx.provider = "udononnxCast"}
  func.func private @__udon_onnx_equal(tensor<1x?xf32>, tensor<f32>) -> tensor<1x?xi1>
      attributes {udon.onnx.operation = "Equal", udon.onnx.provider = "udononnxEqual"}
  func.func private @__udon_onnx_boundedprefix(tensor<1x8xf32>) -> tensor<1x?xf32>
      attributes {udon.onnx.operation = "BoundedPrefix", udon.onnx.provider = "udononnxBoundedPrefix"}
  func.func private @__udon_onnx_where(tensor<1x?xi1>, tensor<f32>, tensor<1x?xf32>) -> tensor<1x?xf32>
      attributes {udon.onnx.operation = "Where", udon.onnx.provider = "udononnxWhere"}
  func.func private @__udoncl_static_tensor_0() -> tensor<f32>
      attributes {udoncl.static_tensor = {alignment = 4 : i64, byte_length = 4 : i64, byte_strides = array<i64>, data = dense<0.0> : tensor<f32>, data_type = "float32", id = 0 : i32, layout = "row_major_contiguous", name = "zero", sha256 = "", shape = array<i64>, symbol = "__udoncl_static_tensor_0"}}
  func.func private @__udoncl_static_tensor_1() -> tensor<f32>
      attributes {udoncl.static_tensor = {alignment = 4 : i64, byte_length = 4 : i64, byte_strides = array<i64>, data = dense<-3.0e+38> : tensor<f32>, data_type = "float32", id = 1 : i32, layout = "row_major_contiguous", name = "minimum", sha256 = "", shape = array<i64>, symbol = "__udoncl_static_tensor_1"}}
  func.func private @__udoncl_static_tensor_2() -> tensor<1x8xf32>
      attributes {udoncl.static_tensor = {alignment = 4 : i64, byte_length = 32 : i64, byte_strides = array<i64: 32, 4>, data = dense<0.0> : tensor<1x8xf32>, data_type = "float32", id = 2 : i32, layout = "row_major_contiguous", name = "causal", sha256 = "", shape = array<i64: 1, 8>, symbol = "__udoncl_static_tensor_2"}}

  func.func @main_graph(
      %mask: tensor<1x?xi64> {onnx.name = "attention_mask", udoncl.dim_exprs = [{axis = 1 : i64, offset = 1 : i64, symbol = "past"}]})
      -> (tensor<1x?xf32> {onnx.name = "padding"}) {
    %zero = func.call @__udoncl_static_tensor_0() : () -> tensor<f32>
    %minimum = func.call @__udoncl_static_tensor_1() : () -> tensor<f32>
    %causal_bound = func.call @__udoncl_static_tensor_2() : () -> tensor<1x8xf32>
    %float = func.call @__udon_onnx_cast(%mask) {udon.onnx.attributes = {to = f32}, udon.onnx.result_dim_expr = {axis = 1 : i64, offset = 1 : i64, symbol = "past"}}
        : (tensor<1x?xi64>) -> tensor<1x?xf32>
    %padded = func.call @__udon_onnx_equal(%float, %zero) {udon.onnx.result_dim_expr = {axis = 1 : i64, offset = 1 : i64, symbol = "past"}}
        : (tensor<1x?xf32>, tensor<f32>) -> tensor<1x?xi1>
    %causal = func.call @__udon_onnx_boundedprefix(%causal_bound) {udon.onnx.attributes = {axis = 1 : i64, offset = 1 : i64, symbol = "past"}, udon.onnx.result_dim_expr = {axis = 1 : i64, offset = 1 : i64, symbol = "past"}}
        : (tensor<1x8xf32>) -> tensor<1x?xf32>
    %padding = func.call @__udon_onnx_where(%padded, %minimum, %causal) {udon.onnx.result_dim_expr = {axis = 1 : i64, offset = 1 : i64, symbol = "past"}}
        : (tensor<1x?xi1>, tensor<f32>, tensor<1x?xf32>) -> tensor<1x?xf32>
    return %padding : tensor<1x?xf32>
  }
}

// CHECK-LABEL: func.func @udononnxGeneratedRun(%arg0
// CHECK: call @clFlush
// CHECK: udonhl.address_of @__udon_onnx_dimension_0
// CHECK-NEXT: %[[PAST:.*]] = "udonhl.load"
// CHECK: %[[EXTENT:.*]] = arith.addi %[[PAST]], %c1_i32
// CHECK: call @__vrc_opencl_dispatch_udononnx_cast_i64_f32(%{{.*}}, %{{.*}}, %[[EXTENT]])
// CHECK: %[[HIGH:.*]] = arith.shli %{{.*}}, %c16_i32
// CHECK: %[[PACKED:.*]] = arith.ori %[[HIGH]], %c1_i32
// CHECK: call @__vrc_opencl_dispatch_udononnx_equal_f32(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %c{{[0-9]+}}_i32, %[[PACKED]],
// CHECK: call @__vrc_opencl_dispatch_udononnx_where_f32(
