// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph --udoncl-materialize-onnx-host-globals --udoncl-materialize-onnx-host-model-registry %s | %FileCheck --implicit-check-not='@udon_onnx_binding_create' --implicit-check-not='udoncl.onnx_binding =' --implicit-check-not='@clEnqueue' %s

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

// CHECK: module attributes {
// CHECK-SAME: udoncl.onnx_host_globals_materialized
// CHECK-SAME: udoncl.onnx_host_model_registry_materialized
// CHECK: func.func private @__udon_onnx_add
// CHECK: func.func @main_graph
// CHECK: sym_name = "__udon_onnx_model_storage"
// CHECK: func.func @udononnxGeneratedGetModelCount() -> i32 {
// CHECK: func.func @udononnxGeneratedGetModel(%{{.*}}: i32, %{{.*}}: !udonhl.ptr<0>) -> i32 {
// CHECK: udonhl.address_of @__udon_onnx_model_storage
// CHECK: udonhl.address_of @__udon_onnx_input_metadata
// CHECK: udonhl.address_of @__udon_onnx_output_metadata
