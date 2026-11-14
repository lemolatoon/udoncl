// expected-error @below {{duplicate configured public Device endpoint}}
module attributes {
  vrc.static_device_resource_inventory = {schema_version = 1 : i32, resources = []},
  vrc.static_public_resource_aliases = {schema_version = 1 : i32, resolved_count = 0 : i32, aliases = []},
  vrc.unity_configured_buffers = {schema_version = 1 : i32, buffers = [
    {buffer_id = 1 : i32, capacity_bytes = 8 : i64, heap_slot = "a", public_access = "input", public_index = 0 : i32},
    {buffer_id = 2 : i32, capacity_bytes = 8 : i64, heap_slot = "b", public_access = "input", public_index = 0 : i32}
  ]}
} {
}

// -----

// expected-error @below {{configured Unity buffer capacity does not match its static public resource}}
module attributes {
  vrc.static_device_resource_inventory = {schema_version = 1 : i32, resources = [
    {capacity_bytes = 16 : i64, format = "rgba8_unorm", public = true, public_access = "output", public_index = 0 : i32, resource_id = 0 : i32, resource_kind = "allocation"}
  ]},
  vrc.static_public_resource_aliases = {schema_version = 1 : i32, resolved_count = 1 : i32, aliases = [
    {public_access = "output", public_index = 0 : i32, resource_id = 0 : i32, resource_kind = "allocation", texture_id = 1 : i32}
  ]},
  vrc.unity_configured_buffers = {schema_version = 1 : i32, buffers = [
    {buffer_id = 1 : i32, capacity_bytes = 8 : i64, heap_slot = "output", public_access = "output", public_index = 0 : i32}
  ]}
} {
}
