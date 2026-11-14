module attributes {
  vrc.static_device_resource_inventory = {schema_version = 1 : i32, resources = [
    {capacity_bytes = 16 : i64, format = "rgba8_unorm", resource_id = 9 : i32, resource_kind = "allocation"}
  ]},
  vrc.unity_texture_assignments = {schema_version = 1 : i32, assignments = []}
} {
  func.func @missing(%buffer: !vrc.buffer) {
    %c1 = arith.constant 1 : i32
    %desc = vrc.dispatch_desc_create %c1 {kernel_id = 7 : i32, scalar_kinds = ["i32"]} : (i32) -> !vrc.dispatch_desc
    // expected-error @below {{static dispatch allocation has no Texture assignment}}
    %event = vrc.dispatch %desc(%buffer) {argument_kinds = ["buffer", "i32"], global_offset = array<i32: 0, 0, 0>, global_size = array<i32: 1, 1, 1>, kernel_id = 7 : i32, local_size = array<i32: 1, 1, 1>, resource_accesses = ["write"], vrc.static_kernel_binding_snapshot = {resources = [{access = "write", kind = "allocation", operand_index = 0 : i32, resource_id = 9 : i32}], schema_version = 1 : i32}} : (!vrc.dispatch_desc, !vrc.buffer) -> !vrc.event
    return
  }
}

// -----

// expected-error @below {{Texture assignment does not match static resource inventory}}
module attributes {
  vrc.static_device_resource_inventory = {schema_version = 1 : i32, resources = [
    {capacity_bytes = 16 : i64, format = "rgba8_unorm", resource_id = 3 : i32, resource_kind = "allocation"}
  ]},
  vrc.unity_texture_assignments = {schema_version = 1 : i32, assignments = [
    {capacity_bytes = 32 : i64, format = "rgba8_unorm", resource_id = 3 : i32, resource_kind = "allocation", texture_id = 1 : i32}
  ]}
} {
  func.func @mismatch(%buffer: !vrc.buffer) {
    %c1 = arith.constant 1 : i32
    %desc = vrc.dispatch_desc_create %c1 {kernel_id = 8 : i32, scalar_kinds = ["i32"]} : (i32) -> !vrc.dispatch_desc
    %event = vrc.dispatch %desc(%buffer) {argument_kinds = ["buffer", "i32"], global_offset = array<i32: 0, 0, 0>, global_size = array<i32: 1, 1, 1>, kernel_id = 8 : i32, local_size = array<i32: 1, 1, 1>, resource_accesses = ["write"], vrc.static_kernel_binding_snapshot = {resources = [{access = "write", kind = "allocation", operand_index = 0 : i32, resource_id = 3 : i32}], schema_version = 1 : i32}} : (!vrc.dispatch_desc, !vrc.buffer) -> !vrc.event
    return
  }
}
