module attributes {
  vrc.static_device_resource_inventory = {schema_version = 1 : i32, resources = [
    {capacity_bytes = 16 : i64, format = "rgba8_unorm", persistent = true, public = true, public_access = "input", resource_id = 0 : i32, resource_kind = "allocation"},
    {capacity_bytes = 32 : i64, format = "rgba8_unorm", persistent = true, public = false, resource_id = 1 : i32, resource_kind = "allocation"}
  ]},
  vrc.unity_texture_assignments = {schema_version = 1 : i32, assignments = [
    {capacity_bytes = 16 : i64, format = "rgba8_unorm", persistent = true, public = true, public_access = "input", resource_id = 0 : i32, resource_kind = "allocation", texture_id = 1 : i32},
    {capacity_bytes = 32 : i64, format = "rgba8_unorm", persistent = true, public = false, resource_id = 1 : i32, resource_kind = "allocation", texture_id = 2 : i32}
  ]},
  vrc.static_public_resource_aliases = {schema_version = 1 : i32, resolved_count = 1 : i32, aliases = [
    {public_access = "input", public_index = 0 : i32, resource_id = 0 : i32, resource_kind = "allocation", texture_id = 1 : i32}
  ]},
  vrc.static_unity_buffer_reconciliation = {schema_version = 1 : i32, buffers = [{}, {}, {}], configured_count = 3 : i32, resolved_alias_count = 1 : i32, unresolved_private_count = 2 : i32, unresolved_public_count = 0 : i32, unused_alias_count = 0 : i32}
} {
}

// -----

module attributes {
  vrc.static_device_resource_inventory = {schema_version = 1 : i32, resources = [
    {capacity_bytes = 64 : i64, format = "rgba8_unorm", persistent = true, public = false, resource_id = 0 : i32, resource_kind = "allocation"}
  ]},
  vrc.unity_texture_assignments = {schema_version = 1 : i32, assignments = [
    {capacity_bytes = 64 : i64, format = "rgba8_unorm", persistent = true, public = false, resource_id = 0 : i32, resource_kind = "allocation", texture_id = 1 : i32}
  ]},
  vrc.static_public_resource_aliases = {schema_version = 1 : i32, resolved_count = 0 : i32, aliases = []},
  vrc.static_unity_buffer_reconciliation = {schema_version = 1 : i32, buffers = [{}, {}], configured_count = 2 : i32, resolved_alias_count = 0 : i32, unresolved_private_count = 1 : i32, unresolved_public_count = 1 : i32, unused_alias_count = 0 : i32}
} {
}
