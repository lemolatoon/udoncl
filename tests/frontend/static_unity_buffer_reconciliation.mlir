module attributes {
  vrc.static_device_resource_inventory = {schema_version = 1 : i32, resources = [
    {capacity_bytes = 32 : i64, format = "rgba8_unorm", public = true, public_access = "input", public_index = 7 : i32, resource_id = 9 : i32, resource_kind = "allocation"},
    {capacity_bytes = 64 : i64, format = "rgba8_unorm", public = true, public_access = "output", public_index = 0 : i32, resource_id = 2 : i32, resource_kind = "allocation"}
  ]},
  vrc.static_public_resource_aliases = {schema_version = 1 : i32, resolved_count = 2 : i32, aliases = [
    {public_access = "input", public_index = 7 : i32, resource_id = 9 : i32, resource_kind = "allocation", texture_id = 1 : i32},
    {public_access = "output", public_index = 0 : i32, resource_id = 2 : i32, resource_kind = "allocation", texture_id = 4 : i32}
  ]},
  vrc.unity_configured_buffers = {schema_version = 1 : i32, buffers = [
    {buffer_id = 3 : i32, capacity_bytes = 32 : i64, heap_slot = "legacy_same_capacity", public_access = "input", public_index = 8 : i32},
    {buffer_id = 1 : i32, capacity_bytes = 32 : i64, heap_slot = "resolved_input", public_access = "input", public_index = 7 : i32},
    {buffer_id = 4 : i32, capacity_bytes = 16 : i64, heap_slot = "private"}
  ]}
} {
}

// -----

module attributes {
  vrc.static_device_resource_inventory = {schema_version = 1 : i32, resources = []},
  vrc.static_public_resource_aliases = {schema_version = 1 : i32, resolved_count = 0 : i32, aliases = []},
  vrc.unity_configured_buffers = {schema_version = 1 : i32, buffers = [
    {buffer_id = 1 : i32, capacity_bytes = 8 : i64, heap_slot = "legacy", public_access = "output"}
  ]}
} {
}
