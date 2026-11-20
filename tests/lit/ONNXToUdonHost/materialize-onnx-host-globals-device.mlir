// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph=fuse-device-nodes=false --udoncl-materialize-onnx-host-globals %s | %FileCheck %s
// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph --udoncl-materialize-onnx-host-globals %s | %FileCheck %s --check-prefix=FUSED

// An intermediate Device result owns a buffer global. Fused into its reader
// it has no buffer, and the plan says so to this pass.

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
  %first = func.call @__udon_onnx_add(%left, %right)
      : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
  %sum = func.call @__udon_onnx_add(%first, %right)
      : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
  return %sum : tensor<1x4xf32>
}

// CHECK: module attributes {
// CHECK-SAME: udoncl.onnx_host_globals_materialized
// CHECK: func.func private @__udon_onnx_add
// CHECK: func.func @main_graph
// CHECK: sym_name = "__udon_onnx_model_storage"
// CHECK: sym_name = "__udon_onnx_device_buffer_2"
// CHECK: sym_name = "__udon_onnx_device_buffer_owned_2"
// CHECK-NOT: sym_name = "__udon_onnx_has_run"
// CHECK-NOT: func.func @udononnxGeneratedGetModelCount
// CHECK-NOT: func.func @udon_onnx_binding_create

// FUSED: udoncl.onnx_host_globals_materialized
// FUSED: sym_name = "__udon_onnx_model_storage"
// FUSED-NOT: sym_name = "__udon_onnx_device_buffer_
