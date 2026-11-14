// RUN: udoncl-opt --vrc-build-static-device-resource-inventory --split-input-file --verify-diagnostics %s

// expected-error @below {{allocation inventory inputs do not cover discovery}}
module attributes {
  vrc.static_device_resource_discovery = {contexts = [{context_id = 0 : i32}], schema_version = 1 : i32},
  vrc.static_device_resource_requirements = {resources = [], schema_version = 1 : i32},
  vrc.static_device_resource_persistence = {resources = [{persistent = false, resource_id = 0 : i32, resource_kind = "allocation"}], schema_version = 1 : i32}
} {}

// -----

// expected-error @below {{ONNX resource inventory requires public classification}}
module attributes {
  udoncl.onnx_binding = {},
  vrc.static_device_resource_discovery = {contexts = [{context_id = 0 : i32}], schema_version = 1 : i32},
  vrc.static_device_resource_requirements = {resources = [{capacity_bytes = 16 : i64, format = "rgba8_unorm", memory_flags = 1 : i64, resource_id = 0 : i32, resource_kind = "allocation", usage = "storage"}], schema_version = 1 : i32},
  vrc.static_device_resource_persistence = {resources = [{persistent = true, resource_id = 0 : i32, resource_kind = "allocation"}], schema_version = 1 : i32}
} {}

// -----

// expected-error @below {{public binding exceeds physical allocation capacity}}
module attributes {
  vrc.static_device_resource_discovery = {contexts = [{context_id = 0 : i32}], schema_version = 1 : i32},
  vrc.static_device_resource_requirements = {resources = [{capacity_bytes = 16 : i64, format = "rgba8_unorm", memory_flags = 1 : i64, resource_id = 0 : i32, resource_kind = "allocation", usage = "storage"}], schema_version = 1 : i32},
  vrc.static_device_resource_persistence = {resources = [{persistent = false, resource_id = 0 : i32, resource_kind = "allocation"}], schema_version = 1 : i32},
  vrc.static_device_resource_public_bindings = {resources = [{capacity_bytes = 32 : i64, data_type = "float32", logical_bytes = 16 : i64, name = "input", public_index = 0 : i32, public_access = "input", resource_id = 0 : i32, resource_kind = "allocation"}], schema_version = 1 : i32}
} {}
