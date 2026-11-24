// RUN: %vrc-opt --split-input-file --vrc-describe-onnx-binding-manifest %s | %FileCheck %s

// A library built with TRANSITIONAL_SEQUENCE_STATE names the symbols of its
// private binding, and a graph with sequence state also its advance and reset
// symbols. A description of an earlier run is replaced.

// CHECK-LABEL: module attributes
// CHECK-SAME: vrc.onnx_binding_manifest = {abi = "udononnx-runtime-v2", inputs = [{capacity_bytes = 64 : i64, data_type = "float32", dimensions = [1, 2, 0, 8], index = 0 : i64, logical_bytes = 0 : i64, name = "past", rank = 4 : i64}], model_index = 0 : i64, model_name = "model_0", outputs = [{capacity_bytes = 64 : i64, data_type = "float32", dimensions = [1, 2, 1, 8], index = 0 : i64, logical_bytes = 64 : i64, name = "present", rank = 4 : i64}], physical = {bind_all_symbol = "b", create_symbol = "c", dispose_symbol = "d", enqueue_advance_sequence_state_symbol = "a", enqueue_initialize_symbol = "i", enqueue_reset_sequence_state_symbol = "z", enqueue_run_symbol = "r", resolve_input_symbol = "ri", resolve_output_symbol = "ro"}}
module attributes {
  udoncl.onnx_binding = {
    abi = "udononnx-runtime-v2", model_index = 0 : i32, model_name = "model_0",
    inputs = [{capacity_bytes = 64 : i64, data_type = "float32", dimensions = array<i64: 1, 2, 0, 8>, index = 0 : i32, logical_bytes = 0 : i64, name = "past", rank = 4 : i32}],
    outputs = [{capacity_bytes = 64 : i64, data_type = "float32", dimensions = array<i64: 1, 2, 1, 8>, index = 0 : i32, logical_bytes = 64 : i64, name = "present", rank = 4 : i32}],
    physical = {bind_all_symbol = "b", create_symbol = "c", dispose_symbol = "d", enqueue_advance_sequence_state_symbol = "a", enqueue_initialize_symbol = "i", enqueue_reset_sequence_state_symbol = "z", enqueue_run_symbol = "r", resolve_input_symbol = "ri", resolve_output_symbol = "ro"}
  },
  vrc.onnx_binding_manifest = {}
} {
}

// -----

// Any other library has no private binding and names no physical symbols.

// CHECK-LABEL: module attributes
// CHECK-SAME: vrc.onnx_binding_manifest = {abi = "udononnx-runtime-v2", inputs = [], model_index = 0 : i64, model_name = "model_0", outputs = []}
module attributes {
  udoncl.onnx_binding = {
    abi = "udononnx-runtime-v2", model_index = 0 : i32, model_name = "model_0",
    inputs = [], outputs = []
  }
} {
}

// -----

// A module without ONNX binding metadata gets no description.

// CHECK-LABEL: module attributes
// CHECK-NOT: vrc.onnx_binding_manifest
// CHECK-SAME: vrc.unrelated
module attributes {
  vrc.onnx_binding_manifest = {},
  vrc.unrelated
} {
}
