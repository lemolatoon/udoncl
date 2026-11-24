// RUN: %vrc-opt --vrc-verify-onnx-session-bindings %s | %FileCheck %s

// Tensors without an element share nothing: a prefill Host binds every empty
// past to one buffer.
// CHECK: vrc.onnx_session_bindings = [{access = "input", index = 0 : i64, resource_id = 0 : i64, tensor = "past_key"}, {access = "input", index = 1 : i64, resource_id = 0 : i64, tensor = "past_value"}]
module attributes {
  udoncl.onnx_binding = {
    inputs = [{capacity_bytes = 0 : i64, data_type = "float32", dimensions = array<i64: 1, 0>, index = 0 : i32, logical_bytes = 0 : i64, name = "past_key", rank = 2 : i32}, {capacity_bytes = 0 : i64, data_type = "float32", dimensions = array<i64: 1, 0>, index = 1 : i32, logical_bytes = 0 : i64, name = "past_value", rank = 2 : i32}],
    outputs = []
  },
  vrc.static_device_global_cell_provenance = {cells = [
    {global = @__udon_onnx_bound_input_0, may_be_null = true, resource_ids = [0 : i32], unknown = false},
    {global = @__udon_onnx_bound_input_1, may_be_null = true, resource_ids = [0 : i32], unknown = false}
  ], schema_version = 1 : i32},
  vrc.static_device_resource_requirements = {resources = [
    {capacity_bytes = 4 : i64, memory_flags = 4 : i64, resource_id = 0 : i32}
  ], schema_version = 1 : i32}
} {
  func.func private @udononnxRun(!udonhl.ptr<0>) -> i32
  func.func @Run(%session: !udonhl.ptr<0>) attributes {udoncl.export} {
    %0 = func.call @udononnxRun(%session) : (!udonhl.ptr<0>) -> i32
    return
  }
}
