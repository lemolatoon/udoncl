// RUN: %vrc-opt --split-input-file --vrc-describe-canonical-unity-buffers %s | %FileCheck %s

// The Unity buffers of an applied lowering, in layout order. A public buffer
// is named after its public name made a C# identifier; a name taken already
// gets the suffix of its access and index. A private buffer is
// private_<texture id>.

// CHECK-LABEL: module attributes
// CHECK-SAME: vrc.canonical_unity_buffers = [{byte_size = 16 : i64, element_type = "float32", id = 1 : i64, logical_byte_size = 12 : i64, name = "x", public_access = "input", public_index = 0 : i64}, {byte_size = 8 : i64, element_type = "int64", id = 2 : i64, logical_byte_size = 8 : i64, name = "tensor_0_logits", public_access = "output", public_index = 0 : i64}, {byte_size = 4 : i64, element_type = "int32", id = 3 : i64, logical_byte_size = 4 : i64, name = "x_output_1", public_access = "output", public_index = 1 : i64}, {byte_size = 32 : i64, id = 4 : i64, name = "private_4"}]
module attributes {
  vrc.static_unity_buffer_layout_plan = {buffers = [
    {buffer_id = 1 : i32, capacity_bytes = 16 : i64, data_type = "float32", format = "rgba8_unorm", heap_slot = "__udoncl_buffer_1", logical_bytes = 12 : i64, name = "x", persistent = true, public = true, public_access = "input", public_index = 0 : i32, resource_id = 0 : i32, texture_id = 1 : i32},
    {buffer_id = 2 : i32, capacity_bytes = 8 : i64, data_type = "int64", format = "rgba8_unorm", heap_slot = "__udoncl_buffer_2", logical_bytes = 8 : i64, name = "0.logits", persistent = true, public = true, public_access = "output", public_index = 0 : i32, resource_id = 3 : i32, texture_id = 2 : i32},
    {buffer_id = 3 : i32, capacity_bytes = 4 : i64, data_type = "int32", format = "rgba8_unorm", heap_slot = "__udoncl_buffer_3", logical_bytes = 4 : i64, name = "x", persistent = true, public = true, public_access = "output", public_index = 1 : i32, resource_id = 1 : i32, texture_id = 3 : i32},
    {buffer_id = 4 : i32, capacity_bytes = 32 : i64, format = "rgba8_unorm", heap_slot = "__udoncl_buffer_4", persistent = true, public = false, resource_id = 2 : i32, texture_id = 4 : i32}
  ], ready = true, schema_version = 1 : i32},
  vrc.static_unity_resource_lowering_readiness = {applied = true, pool_replacement_ready = true, schema_version = 1 : i32}
} {
}

// -----

// A lowering that is not applied describes no buffers, and a description of
// an earlier run does not survive.

// CHECK-LABEL: module attributes
// CHECK-NOT: vrc.canonical_unity_buffers
// CHECK-SAME: vrc.static_unity_resource_lowering_readiness
module attributes {
  vrc.canonical_unity_buffers = [],
  vrc.static_unity_resource_lowering_readiness = {applied = false, pool_replacement_ready = false, schema_version = 1 : i32}
} {
}

// -----

// A public buffer of a UDONCL_PUBLIC Host variable comes from that variable.
// A buffer the Host source does not declare states no source, even when the
// ONNX binding has a graph tensor of its name: only a session binding ties a
// buffer to a tensor.

// CHECK-LABEL: module attributes
// CHECK-SAME: vrc.canonical_unity_buffers = [{byte_size = 4 : i64, element_type = "float32", id = 1 : i64, logical_byte_size = 4 : i64, name = "image", public_access = "input", public_index = 0 : i64, source = {kind = "host_variable", symbol = "image"}}, {byte_size = 4 : i64, element_type = "float32", id = 2 : i64, logical_byte_size = 4 : i64, name = "logits", public_access = "output", public_index = 0 : i64}, {byte_size = 4 : i64, element_type = "float32", id = 3 : i64, logical_byte_size = 4 : i64, name = "hidden", public_access = "output", public_index = 1 : i64}]
module attributes {
  udoncl.onnx_binding = {inputs = [{index = 0 : i32, name = "input"}], outputs = [{index = 0 : i32, name = "logits"}, {index = 2 : i32, name = "hidden"}]},
  vrc.static_declared_public_resources = {resource_ids = array<i32: 4>, schema_version = 1 : i32},
  vrc.static_unity_buffer_layout_plan = {buffers = [
    {buffer_id = 1 : i32, capacity_bytes = 4 : i64, data_type = "float32", format = "rgba8_unorm", heap_slot = "__udoncl_buffer_1", logical_bytes = 4 : i64, name = "image", persistent = true, public = true, public_access = "input", public_index = 0 : i32, resource_id = 4 : i32, texture_id = 1 : i32},
    {buffer_id = 2 : i32, capacity_bytes = 4 : i64, data_type = "float32", format = "rgba8_unorm", heap_slot = "__udoncl_buffer_2", logical_bytes = 4 : i64, name = "logits", persistent = true, public = true, public_access = "output", public_index = 0 : i32, resource_id = 5 : i32, texture_id = 2 : i32},
    {buffer_id = 3 : i32, capacity_bytes = 4 : i64, data_type = "float32", format = "rgba8_unorm", heap_slot = "__udoncl_buffer_3", logical_bytes = 4 : i64, name = "hidden", persistent = true, public = true, public_access = "output", public_index = 1 : i32, resource_id = 6 : i32, texture_id = 3 : i32}
  ], ready = true, schema_version = 1 : i32},
  vrc.static_unity_resource_lowering_readiness = {applied = true, pool_replacement_ready = true, schema_version = 1 : i32}
} {
}

// -----

// A public buffer U# both writes and reads has the access inout; a taken name
// gets the inout suffix.

// CHECK-LABEL: module attributes
// CHECK-SAME: vrc.canonical_unity_buffers = [{byte_size = 4 : i64, element_type = "float32", id = 1 : i64, logical_byte_size = 4 : i64, name = "state", public_access = "input", public_index = 0 : i64}, {byte_size = 4 : i64, element_type = "float32", id = 2 : i64, logical_byte_size = 4 : i64, name = "state_inout_0", public_access = "inout", public_index = 0 : i64}]
module attributes {
  udoncl.onnx_binding = {inputs = [{index = 0 : i32, name = "state"}], outputs = [{index = 0 : i32, name = "state"}]},
  vrc.static_unity_buffer_layout_plan = {buffers = [
    {buffer_id = 1 : i32, capacity_bytes = 4 : i64, data_type = "float32", format = "rgba8_unorm", heap_slot = "__udoncl_buffer_1", logical_bytes = 4 : i64, name = "state", persistent = true, public = true, public_access = "input", public_index = 0 : i32, resource_id = 0 : i32, texture_id = 1 : i32},
    {buffer_id = 2 : i32, capacity_bytes = 4 : i64, data_type = "float32", format = "rgba8_unorm", heap_slot = "__udoncl_buffer_2", logical_bytes = 4 : i64, name = "state", persistent = true, public = true, public_access = "inout", public_index = 0 : i32, resource_id = 1 : i32, texture_id = 2 : i32}
  ], ready = true, schema_version = 1 : i32},
  vrc.static_unity_resource_lowering_readiness = {applied = true, pool_replacement_ready = true, schema_version = 1 : i32}
} {
}
