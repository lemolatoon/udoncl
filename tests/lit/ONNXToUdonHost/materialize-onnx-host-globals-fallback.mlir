// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph --udoncl-materialize-onnx-host-globals %s | %FileCheck %s

func.func private @__udon_onnx_add(
    tensor<1x4xi32>, tensor<1x4xi32>) -> tensor<1x4xi32>
    attributes {
      udon.onnx.operation = "Add",
      udon.onnx.provider = "udonatenAdd"
    }

func.func @main_graph(
    %left: tensor<1x4xi32> {onnx.name = "left"},
    %right: tensor<1x4xi32> {onnx.name = "right"})
    -> (tensor<1x4xi32> {onnx.name = "output"}) {
  %output = func.call @__udon_onnx_add(%left, %right)
      : (tensor<1x4xi32>, tensor<1x4xi32>) -> tensor<1x4xi32>
  return %output : tensor<1x4xi32>
}

// CHECK: module attributes {
// CHECK-SAME: udoncl.onnx_host_globals_materialized
// CHECK-SAME: device_graph = false
// CHECK: func.func private @__udon_onnx_add
// CHECK: func.func @main_graph
// CHECK: sym_name = "__udon_onnx_input_0"
// CHECK: sym_name = "__udon_onnx_value_2"
// CHECK: sym_name = "__udon_onnx_input_binding_0"
// CHECK: sym_name = "__udon_onnx_input_binding_1"
// CHECK: sym_name = "__udon_onnx_has_run"
// CHECK-NOT: func.func @udononnxGeneratedGetModelCount
// CHECK-NOT: func.func @udon_onnx_binding_create
