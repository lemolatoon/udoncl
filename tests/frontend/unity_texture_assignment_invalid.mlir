// expected-error @below {{duplicate keyed static resource identity}}
module attributes {
  vrc.static_device_resource_discovery = {contexts = [{context_id = 0 : i32}], schema_version = 1 : i32},
  vrc.static_device_resource_inventory = {resources = [
    {capacity_bytes = 16 : i64, format = "rgba8_unorm", persistent = false, public = false, resource_id = 0 : i32, resource_kind = "allocation"},
    {capacity_bytes = 16 : i64, format = "rgba8_unorm", persistent = false, public = false, resource_id = 0 : i32, resource_kind = "allocation"}
  ], schema_version = 1 : i32}
} {}

// -----

// expected-error @below {{invalid allocation Texture requirements}}
module attributes {
  vrc.static_device_resource_discovery = {contexts = [{context_id = 0 : i32}], schema_version = 1 : i32},
  vrc.static_device_resource_inventory = {resources = [
    {capacity_bytes = 0 : i64, format = "rgba8_unorm", persistent = false, public = false, resource_id = 0 : i32, resource_kind = "allocation"}
  ], schema_version = 1 : i32}
} {}

// -----

// expected-error @below {{invalid allocation Texture requirements}}
module attributes {
  vrc.static_device_resource_discovery = {contexts = [{context_id = 0 : i32}], schema_version = 1 : i32},
  vrc.static_device_resource_inventory = {resources = [
    {capacity_bytes = 16 : i64, format = "r32_float", persistent = false, public = false, resource_id = 0 : i32, resource_kind = "allocation"}
  ], schema_version = 1 : i32}
} {}

// -----

// expected-error @below {{allocation inventory references unknown context}}
module attributes {
  vrc.static_device_resource_discovery = {contexts = [{context_id = 0 : i32}], schema_version = 1 : i32},
  vrc.static_device_resource_inventory = {resources = [
    {capacity_bytes = 16 : i64, format = "rgba8_unorm", persistent = false, public = false, resource_id = 1 : i32, resource_kind = "allocation"}
  ], schema_version = 1 : i32}
} {}
