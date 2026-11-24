// RUN: %vrc-opt --split-input-file --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph --udoncl-materialize-onnx-host-globals %s | %FileCheck %s

// A library built with TRANSITIONAL_SEQUENCE_STATE (udoncl.onnx_sequence_states,
// here without a state) gets the state of its private transitional binding.
module attributes {udoncl.onnx_sequence_states = []} {
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
}

// CHECK-LABEL: udoncl.onnx_sequence_states
// CHECK-DAG: sym_name = "__udon_onnx_model_storage"
// CHECK-DAG: sym_name = "__udon_onnx_binding_state"
// CHECK-DAG: sym_name = "__udon_onnx_binding_inputs"
// CHECK-DAG: sym_name = "__udon_onnx_binding_outputs"
// CHECK-DAG: sym_name = "__udon_onnx_io_handles"
// CHECK-DAG: sym_name = "__udon_onnx_binding_event"

// -----

// Any other library binds its buffers through the UdonONNX session API and
// has no binding state.
module attributes {test.session_api_host} {
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
}

// CHECK-LABEL: test.session_api_host
// CHECK-NOT: __udon_onnx_binding_
// CHECK-NOT: __udon_onnx_io_handles
// CHECK: sym_name = "__udon_onnx_model_storage"
// CHECK-NOT: __udon_onnx_binding_
// CHECK-NOT: __udon_onnx_io_handles
