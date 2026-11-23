// RUN: %vrc-opt --split-input-file --verify-diagnostics --vrc-describe-canonical-unity-buffers %s

// An applied lowering requires a ready canonical layout.

// expected-error @below {{applied static Unity lowering has no canonical buffer layout}}
module attributes {
  vrc.static_unity_resource_lowering_readiness = {applied = true, pool_replacement_ready = true, schema_version = 1 : i32}
} {
}

// -----

// A buffer's id is its texture id, and its heap slot is named after it.

// expected-error @below {{invalid canonical Unity buffer layout entry}}
module attributes {
  vrc.static_unity_buffer_layout_plan = {buffers = [
    {buffer_id = 1 : i32, capacity_bytes = 4 : i64, format = "rgba8_unorm", heap_slot = "__udoncl_buffer_2", persistent = true, public = false, resource_id = 0 : i32, texture_id = 1 : i32}
  ], ready = true, schema_version = 1 : i32},
  vrc.static_unity_resource_lowering_readiness = {applied = true, pool_replacement_ready = true, schema_version = 1 : i32}
} {
}

// -----

// A public buffer states its element type, logical size, access and index.

// expected-error @below {{invalid public canonical Unity buffer layout entry}}
module attributes {
  vrc.static_unity_buffer_layout_plan = {buffers = [
    {buffer_id = 1 : i32, capacity_bytes = 4 : i64, data_type = "float64", format = "rgba8_unorm", heap_slot = "__udoncl_buffer_1", logical_bytes = 4 : i64, name = "x", persistent = true, public = true, public_access = "input", public_index = 0 : i32, resource_id = 0 : i32, texture_id = 1 : i32}
  ], ready = true, schema_version = 1 : i32},
  vrc.static_unity_resource_lowering_readiness = {applied = true, pool_replacement_ready = true, schema_version = 1 : i32}
} {
}

// -----

// expected-error @below {{private canonical Unity buffer has public metadata}}
module attributes {
  vrc.static_unity_buffer_layout_plan = {buffers = [
    {buffer_id = 1 : i32, capacity_bytes = 4 : i64, format = "rgba8_unorm", heap_slot = "__udoncl_buffer_1", name = "x", persistent = true, public = false, resource_id = 0 : i32, texture_id = 1 : i32}
  ], ready = true, schema_version = 1 : i32},
  vrc.static_unity_resource_lowering_readiness = {applied = true, pool_replacement_ready = true, schema_version = 1 : i32}
} {
}

// -----

// A public name and its suffixed form are both taken.

// expected-error @below {{canonical Unity buffer names are not uniquely mappable}}
module attributes {
  vrc.static_unity_buffer_layout_plan = {buffers = [
    {buffer_id = 1 : i32, capacity_bytes = 4 : i64, data_type = "float32", format = "rgba8_unorm", heap_slot = "__udoncl_buffer_1", logical_bytes = 4 : i64, name = "x_input_1", persistent = true, public = true, public_access = "input", public_index = 0 : i32, resource_id = 0 : i32, texture_id = 1 : i32},
    {buffer_id = 2 : i32, capacity_bytes = 4 : i64, data_type = "float32", format = "rgba8_unorm", heap_slot = "__udoncl_buffer_2", logical_bytes = 4 : i64, name = "x", persistent = true, public = true, public_access = "output", public_index = 0 : i32, resource_id = 1 : i32, texture_id = 2 : i32},
    {buffer_id = 3 : i32, capacity_bytes = 4 : i64, data_type = "float32", format = "rgba8_unorm", heap_slot = "__udoncl_buffer_3", logical_bytes = 4 : i64, name = "x", persistent = true, public = true, public_access = "input", public_index = 1 : i32, resource_id = 2 : i32, texture_id = 3 : i32}
  ], ready = true, schema_version = 1 : i32},
  vrc.static_unity_resource_lowering_readiness = {applied = true, pool_replacement_ready = true, schema_version = 1 : i32}
} {
}

// -----

// A public buffer already took the name of a private buffer.

// expected-error @below {{canonical Unity buffer names are not unique}}
module attributes {
  vrc.static_unity_buffer_layout_plan = {buffers = [
    {buffer_id = 1 : i32, capacity_bytes = 4 : i64, data_type = "float32", format = "rgba8_unorm", heap_slot = "__udoncl_buffer_1", logical_bytes = 4 : i64, name = "private_2", persistent = true, public = true, public_access = "input", public_index = 0 : i32, resource_id = 0 : i32, texture_id = 1 : i32},
    {buffer_id = 2 : i32, capacity_bytes = 4 : i64, format = "rgba8_unorm", heap_slot = "__udoncl_buffer_2", persistent = true, public = false, resource_id = 1 : i32, texture_id = 2 : i32}
  ], ready = true, schema_version = 1 : i32},
  vrc.static_unity_resource_lowering_readiness = {applied = true, pool_replacement_ready = true, schema_version = 1 : i32}
} {
}

// -----

// expected-error @below {{canonical Unity Texture IDs must be dense from one}}
module attributes {
  vrc.static_unity_buffer_layout_plan = {buffers = [
    {buffer_id = 2 : i32, capacity_bytes = 4 : i64, format = "rgba8_unorm", heap_slot = "__udoncl_buffer_2", persistent = true, public = false, resource_id = 0 : i32, texture_id = 2 : i32}
  ], ready = true, schema_version = 1 : i32},
  vrc.static_unity_resource_lowering_readiness = {applied = true, pool_replacement_ready = true, schema_version = 1 : i32}
} {
}
