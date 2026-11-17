// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph %s | %FileCheck %s

func.func private @__udon_onnx_add(
    tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
    attributes {
      udon.onnx.operation = "Add",
      udon.onnx.provider = "udonatenAdd"
    }

func.func @main_graph(
    %left: tensor<1x4xf32> {onnx.name = "left"},
    %right: tensor<1x4xf32> {onnx.name = "right"})
    -> (tensor<1x4xf32> {onnx.name = "sum"}) {
  %sum = func.call @__udon_onnx_add(%left, %right)
      : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
  return %sum : tensor<1x4xf32>
}

// CHECK: udoncl.onnx_host_graph_validated
// CHECK: udoncl.onnx_host_plan = {
// CHECK-SAME: device_graph = true
// CHECK-SAME: device_storage = array<i64:
// CHECK-SAME: mutable_buffer_bytes = array<i64:
// CHECK-SAME: mutable_handle_capacity =
// CHECK-SAME: operator_scratch_bytes = 0 : i64
// CHECK-SAME: owned_device_buffers = array<i64
// CHECK: func.func @main_graph
