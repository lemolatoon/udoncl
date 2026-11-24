// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph --udoncl-materialize-onnx-host-globals --udoncl-materialize-onnx-host-model-registry %s | %FileCheck --implicit-check-not='udoncl.onnx_binding =' --implicit-check-not='@clEnqueue' %s

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

// The registry is initializer data. The model record is {name, graph id 0,
// inputs, 2, outputs, 1}; each metadata row is {name, FLOAT32 = 6, rank 2,
// dimensions, no bounds}. Every pointer field is a relocation.
// CHECK: "udonhl.global"()
// CHECK-SAME: initial_value = array<i8: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
// CHECK-SAME: relocations = [{addend = 0 : i32, offset = 0 : i32, symbol = @__udon_onnx_model_name}, {addend = 0 : i32, offset = 8 : i32, symbol = @__udon_onnx_input_metadata}, {addend = 0 : i32, offset = 16 : i32, symbol = @__udon_onnx_output_metadata}]
// CHECK-SAME: sym_name = "__udon_onnx_model_storage"
// CHECK: "udonhl.global"()
// CHECK-SAME: initial_value = array<i8: 0, 0, 0, 0, 6, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 2, 0, 0, 0, 0,
// CHECK-SAME: relocations = [{addend = 0 : i32, offset = 0 : i32, symbol = @__udon_onnx_input_name_0}, {addend = 0 : i32, offset = 12 : i32, symbol = @__udon_onnx_input_dimensions_0}, {addend = 0 : i32, offset = 20 : i32, symbol = @__udon_onnx_input_name_1}, {addend = 0 : i32, offset = 32 : i32, symbol = @__udon_onnx_input_dimensions_1}]
// CHECK-SAME: sym_name = "__udon_onnx_input_metadata"
// CHECK: "udonhl.global"()
// CHECK-SAME: initial_value = array<i8: 0, 0, 0, 0, 6, 0, 0, 0, 2, 0, 0, 0, 0,
// CHECK-SAME: relocations = [{addend = 0 : i32, offset = 0 : i32, symbol = @__udon_onnx_output_name_0}, {addend = 0 : i32, offset = 12 : i32, symbol = @__udon_onnx_output_dimensions_0}]
// CHECK-SAME: sym_name = "__udon_onnx_output_metadata"
// CHECK: func.func @udononnxGeneratedGetModelCount() -> i32 {

// Looking the model up stores only the result.
// CHECK: func.func @udononnxGeneratedGetModel(%{{.*}}: i32, %[[OUTPUT:.*]]: !udonhl.ptr<0>) -> i32 {
// CHECK: %[[MODEL:.*]] = udonhl.address_of @__udon_onnx_model_storage
// CHECK-NEXT: "udonhl.store"(%[[MODEL]], %[[OUTPUT]])
// CHECK-NOT: udonhl.store
