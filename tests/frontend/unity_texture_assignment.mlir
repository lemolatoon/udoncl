module attributes {
  vrc.static_device_resource_discovery = {
    contexts = [
      {context_id = 0 : i32},
      {context_id = 1 : i32},
      {context_id = 2 : i32}
    ],
    schema_version = 1 : i32
  },
  vrc.static_device_resource_inventory = {
    resources = [
      {capacity_bytes = 64 : i64, format = "rgba8_unorm", persistent = false, public = false, resource_id = 2 : i32, resource_kind = "allocation"},
      {resource_id = 0 : i32, resource_kind = "static_tensor"},
      {capacity_bytes = 16 : i64, format = "rgba8_unorm", persistent = true, public = true, public_access = "input", resource_id = 0 : i32, resource_kind = "allocation"},
      {capacity_bytes = 32 : i64, format = "rgba8_unorm", persistent = true, public = true, public_access = "output", resource_id = 1 : i32, resource_kind = "allocation"}
    ],
    schema_version = 1 : i32
  }
} {
}
