module attributes {
  udoncl.onnx_binding = {},
  vrc.static_device_resource_discovery = {contexts = [{context_id = 9 : i32}, {context_id = 2 : i32}, {context_id = 5 : i32}], schema_version = 1 : i32},
  vrc.static_device_resource_requirements = {resources = [
    {capacity_bytes = 96 : i64, format = "rgba8_unorm", memory_flags = 1 : i64, resource_id = 9 : i32, resource_kind = "allocation", usage = "storage"},
    {capacity_bytes = 16 : i64, format = "rgba8_unorm", memory_flags = 4 : i64, resource_id = 2 : i32, resource_kind = "allocation", usage = "storage"},
    {capacity_bytes = 64 : i64, format = "rgba8_unorm", memory_flags = 2 : i64, resource_id = 5 : i32, resource_kind = "allocation", usage = "storage"}
  ], schema_version = 1 : i32},
  vrc.static_device_resource_persistence = {resources = [
    {persistent = true, resource_id = 9 : i32, resource_kind = "allocation"},
    {persistent = false, resource_id = 5 : i32, resource_kind = "allocation"},
    {persistent = false, resource_id = 2 : i32, resource_kind = "allocation"}
  ], schema_version = 1 : i32},
  vrc.static_device_resource_public_bindings = {resources = [
    {capacity_bytes = 32 : i64, data_type = "int64", logical_bytes = 0 : i64, name = "tokens", public_index = 1 : i32, public_access = "input", resource_id = 5 : i32, resource_kind = "allocation"}
  ], schema_version = 1 : i32}
} {
}

// -----

module attributes {
  vrc.static_device_resource_discovery = {contexts = [{context_id = 4 : i32}], schema_version = 1 : i32},
  vrc.static_device_resource_requirements = {resources = [
    {capacity_bytes = 20 : i64, format = "rgba8_unorm", memory_flags = 1 : i64, resource_id = 4 : i32, resource_kind = "allocation", usage = "storage"}
  ], schema_version = 1 : i32},
  vrc.static_device_resource_persistence = {resources = [
    {persistent = false, resource_id = 4 : i32, resource_kind = "allocation"}
  ], schema_version = 1 : i32}
} {
}
