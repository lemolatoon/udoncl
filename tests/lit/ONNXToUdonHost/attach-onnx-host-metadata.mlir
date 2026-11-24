// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph --udoncl-attach-onnx-host-metadata %s | %FileCheck --implicit-check-not='@udon_onnx_session_run' --implicit-check-not='@__udononnx_transitional_create' --implicit-check-not='udoncl.onnx_host_finalized' %s
// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph --udoncl-attach-onnx-host-metadata --vrc-describe-onnx-binding-manifest %s | %FileCheck --check-prefix=MANIFEST %s

func.func private @__udon_onnx_add(
    tensor<1x4xi32>, tensor<1x4xi32>) -> tensor<1x4xi32>
    attributes {
      udon.onnx.operation = "Add",
      udon.onnx.provider = "udonatenAdd"
    }

func.func @main_graph(
    %left: tensor<1x4xi32> {onnx.name = "left"},
    %right: tensor<1x4xi32> {onnx.name = "right"})
    -> (tensor<1x4xi32> {onnx.name = "sum"}) {
  %sum = func.call @__udon_onnx_add(%left, %right)
      : (tensor<1x4xi32>, tensor<1x4xi32>) -> tensor<1x4xi32>
  return %sum : tensor<1x4xi32>
}

// CHECK: module attributes {
// CHECK-SAME: udoncl.mutable_buffer_bytes = array<i64: 16, 16, 16>
// CHECK-SAME: udoncl.mutable_handle_capacity
// CHECK-SAME: udoncl.onnx_binding = {abi = "udononnx-runtime-v2"
// CHECK-SAME: inputs = [{capacity_bytes = 16 : i64, data_type = "int32", dimensions = array<i64: 1, 4>, index = 0 : i32, logical_bytes = 16 : i64, name = "left", rank = 2 : i32}
// CHECK-SAME: outputs = [{capacity_bytes = 16 : i64, data_type = "int32", dimensions = array<i64: 1, 4>, index = 0 : i32, logical_bytes = 16 : i64, name = "sum", rank = 2 : i32}]}
// CHECK-SAME: udoncl.onnx_device_graph = false
// CHECK-SAME: udoncl.onnx_device_unsupported = [{count = 1 : i64, operation = "Add", provider = "udonatenAdd"}]
// CHECK-SAME: udoncl.onnx_host_metadata_attached
// CHECK-SAME: udoncl.onnx_lowering = "library-calls-v2"
// CHECK-SAME: udoncl.onnx_operator_counts = {Add = 1 : i64}
// CHECK: func.func @main_graph

// The attached metadata describes the manifest entry of the ONNX binding. A
// library built without TRANSITIONAL_SEQUENCE_STATE has no private binding,
// so the entry names no physical symbols.

// MANIFEST: vrc.onnx_binding_manifest = {abi = "udononnx-runtime-v2", inputs = [{capacity_bytes = 16 : i64, data_type = "int32", dimensions = [1, 4], index = 0 : i64, logical_bytes = 16 : i64, name = "left", rank = 2 : i64}, {capacity_bytes = 16 : i64, data_type = "int32", dimensions = [1, 4], index = 1 : i64, logical_bytes = 16 : i64, name = "right", rank = 2 : i64}], model_index = 0 : i64, model_name = "model_0", outputs = [{capacity_bytes = 16 : i64, data_type = "int32", dimensions = [1, 4], index = 0 : i64, logical_bytes = 16 : i64, name = "sum", rank = 2 : i64}]}
