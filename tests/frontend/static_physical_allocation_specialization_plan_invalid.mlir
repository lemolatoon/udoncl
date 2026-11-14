// expected-error @below {{Texture assignment does not match persistent allocation inventory}}
module attributes {
  vrc.static_device_resource_inventory = {resources = [{capacity_bytes = 16 : i64, format = "rgba8_unorm", persistent = true, resource_id = 0 : i32, resource_kind = "allocation"}], schema_version = 1 : i32},
  vrc.unity_texture_assignments = {assignments = [{capacity_bytes = 32 : i64, format = "rgba8_unorm", persistent = true, resource_id = 0 : i32, resource_kind = "allocation", texture_id = 1 : i32}], schema_version = 1 : i32}
} {
}
