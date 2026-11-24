// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph --udoncl-materialize-onnx-host-globals --udoncl-materialize-onnx-host-model-registry %s | %FileCheck --implicit-check-not='@udon_onnx_session_run' --implicit-check-not='udoncl.onnx_binding =' %s

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
// CHECK-SAME: udoncl.onnx_host_model_registry_materialized
// CHECK-SAME: device_graph = false
// The data type of each metadata row is initializer data, not a store.
// CHECK: initial_value = array<i8: 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0,
// CHECK-SAME: sym_name = "__udon_onnx_input_metadata"
// CHECK: sym_name = "__udon_onnx_has_run"
// CHECK: func.func @udononnxGeneratedGetModelCount() -> i32 {
// CHECK: func.func @udononnxGeneratedGetModel
// CHECK-NOT: arith.constant 2 : i32
