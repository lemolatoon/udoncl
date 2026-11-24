// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph=fuse-device-nodes=false %s | %FileCheck %s

// The session allocates only the Device buffers it owns, here the
// intermediate sum; the graph inputs and the output are buffers the caller
// binds and plans itself, so they take no mutable handle of the session.

func.func private @__udon_onnx_add(
    tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
    attributes {
      udon.onnx.operation = "Add",
      udon.onnx.provider = "udonatenAdd"
    }

func.func @main_graph(
    %left: tensor<1x4xf32> {onnx.name = "left"},
    %right: tensor<1x4xf32> {onnx.name = "right"})
    -> (tensor<1x4xf32> {onnx.name = "total"}) {
  %sum = func.call @__udon_onnx_add(%left, %right)
      : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
  %total = func.call @__udon_onnx_add(%sum, %right)
      : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
  return %total : tensor<1x4xf32>
}

// CHECK: udoncl.onnx_host_plan = {
// CHECK-SAME: device_graph = true
// CHECK-SAME: mutable_buffer_bytes = array<i64: 16>
// CHECK-SAME: mutable_handle_capacity = 1 : i64
// CHECK-SAME: owned_device_buffers = array<i64: 2>
