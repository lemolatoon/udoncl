module attributes {
  vrc.static_device_resource_inventory = {schema_version = 1 : i32, resources = [
    {capacity_bytes = 64 : i64, format = "rgba8_unorm", public = true, public_access = "output", public_index = 0 : i32, resource_id = 2 : i32, resource_kind = "allocation"},
    {capacity_bytes = 16 : i64, format = "rgba8_unorm", public = false, resource_id = 5 : i32, resource_kind = "allocation"},
    {capacity_bytes = 32 : i64, format = "rgba8_unorm", public = true, public_access = "input", public_index = 7 : i32, resource_id = 9 : i32, resource_kind = "allocation"},
    {resource_id = 0 : i32, resource_kind = "static_tensor"}
  ]},
  vrc.unity_texture_assignments = {schema_version = 1 : i32, assignments = [
    {capacity_bytes = 16 : i64, format = "rgba8_unorm", persistent = false, public = false, resource_id = 5 : i32, resource_kind = "allocation", texture_id = 3 : i32},
    {capacity_bytes = 32 : i64, format = "rgba8_unorm", persistent = true, public = true, public_access = "input", resource_id = 9 : i32, resource_kind = "allocation", texture_id = 1 : i32},
    {capacity_bytes = 64 : i64, format = "rgba8_unorm", persistent = true, public = true, public_access = "output", resource_id = 2 : i32, resource_kind = "allocation", texture_id = 4 : i32}
  ]}
} {
}

// -----

module attributes {
  vrc.static_device_resource_inventory = {schema_version = 1 : i32, resources = [
    {capacity_bytes = 8 : i64, format = "rgba8_unorm", public = false, resource_id = 0 : i32, resource_kind = "allocation"}
  ]},
  vrc.unity_texture_assignments = {schema_version = 1 : i32, assignments = [
    {capacity_bytes = 8 : i64, format = "rgba8_unorm", persistent = false, public = false, resource_id = 0 : i32, resource_kind = "allocation", texture_id = 1 : i32}
  ]}
} {
}
