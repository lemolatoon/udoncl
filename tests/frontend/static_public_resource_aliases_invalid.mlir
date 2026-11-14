// expected-error @+1 {{public allocation has no Texture assignment}}
module attributes {
  vrc.static_device_resource_inventory = {schema_version = 1 : i32, resources = [
    {capacity_bytes = 16 : i64, format = "rgba8_unorm", public = true, public_access = "input", public_index = 0 : i32, resource_id = 4 : i32, resource_kind = "allocation"}
  ]},
  vrc.unity_texture_assignments = {schema_version = 1 : i32, assignments = []}
} {
}

// -----

// expected-error @+1 {{duplicate public Device endpoint}}
module attributes {
  vrc.static_device_resource_inventory = {schema_version = 1 : i32, resources = [
    {capacity_bytes = 16 : i64, format = "rgba8_unorm", public = true, public_access = "input", public_index = 3 : i32, resource_id = 4 : i32, resource_kind = "allocation"},
    {capacity_bytes = 32 : i64, format = "rgba8_unorm", public = true, public_access = "input", public_index = 3 : i32, resource_id = 5 : i32, resource_kind = "allocation"}
  ]},
  vrc.unity_texture_assignments = {schema_version = 1 : i32, assignments = [
    {capacity_bytes = 16 : i64, format = "rgba8_unorm", persistent = true, public = true, public_access = "input", resource_id = 4 : i32, resource_kind = "allocation", texture_id = 1 : i32},
    {capacity_bytes = 32 : i64, format = "rgba8_unorm", persistent = true, public = true, public_access = "input", resource_id = 5 : i32, resource_kind = "allocation", texture_id = 2 : i32}
  ]}
} {
}

// -----

// expected-error @+1 {{public allocation and Texture assignment are inconsistent}}
module attributes {
  vrc.static_device_resource_inventory = {schema_version = 1 : i32, resources = [
    {capacity_bytes = 16 : i64, format = "rgba8_unorm", public = true, public_access = "output", public_index = 0 : i32, resource_id = 8 : i32, resource_kind = "allocation"}
  ]},
  vrc.unity_texture_assignments = {schema_version = 1 : i32, assignments = [
    {capacity_bytes = 32 : i64, format = "rgba8_unorm", persistent = true, public = true, public_access = "output", resource_id = 8 : i32, resource_kind = "allocation", texture_id = 1 : i32}
  ]}
} {
}
