module attributes {
  vrc.static_device_resource_inventory = {schema_version = 1 : i32, resources = [
    {capacity_bytes = 32 : i64, format = "rgba8_unorm", resource_id = 9 : i32, resource_kind = "allocation"},
    {capacity_bytes = 64 : i64, format = "rgba8_unorm", resource_id = 2 : i32, resource_kind = "allocation"}
  ]},
  vrc.unity_texture_assignments = {schema_version = 1 : i32, assignments = [
    {capacity_bytes = 64 : i64, format = "rgba8_unorm", resource_id = 2 : i32, resource_kind = "allocation", texture_id = 2 : i32},
    {capacity_bytes = 32 : i64, format = "rgba8_unorm", resource_id = 9 : i32, resource_kind = "allocation", texture_id = 1 : i32}
  ]}
} {
  func.func @run(%a: !vrc.buffer, %b: !vrc.buffer, %dynamic: !vrc.buffer) {
    %c1 = arith.constant 1 : i32
    %static = "vrc.static_buffer"() <{byte_size = 16 : i32, resource_id = 4 : i32}> : () -> !vrc.buffer
    %d0 = vrc.dispatch_desc_create %c1 {kernel_id = 7 : i32, scalar_kinds = ["i32"]} : (i32) -> !vrc.dispatch_desc
    %e0 = vrc.dispatch %d0(%a, %static, %b) {argument_kinds = ["buffer", "buffer", "buffer", "i32"], global_offset = array<i32: 0, 0, 0>, global_size = array<i32: 1, 1, 1>, kernel_id = 7 : i32, local_size = array<i32: 1, 1, 1>, resource_accesses = ["read", "read", "write"], vrc.static_kernel_binding_snapshot = {kernel_id = 7 : i32, resources = [{access = "read", kind = "allocation", operand_index = 0 : i32, resource_id = 9 : i32}, {access = "read", kind = "static_tensor", operand_index = 1 : i32, resource_id = 4 : i32}, {access = "write", kind = "allocation", operand_index = 2 : i32, resource_id = 2 : i32}], scalars = [], schema_version = 1 : i32}} : (!vrc.dispatch_desc, !vrc.buffer, !vrc.buffer, !vrc.buffer) -> !vrc.event
    %d1 = vrc.dispatch_desc_create %c1 {kernel_id = 8 : i32, scalar_kinds = ["i32"]} : (i32) -> !vrc.dispatch_desc
    %e1 = vrc.dispatch %d1(%a, %b) {argument_kinds = ["buffer", "buffer", "i32"], global_offset = array<i32: 0, 0, 0>, global_size = array<i32: 1, 1, 1>, kernel_id = 8 : i32, local_size = array<i32: 1, 1, 1>, resource_accesses = ["read", "write"], vrc.static_kernel_binding_snapshot = {kernel_id = 8 : i32, resources = [{access = "read", kind = "allocation", operand_index = 0 : i32, resource_id = 9 : i32}, {access = "write", kind = "allocation", operand_index = 1 : i32, resource_id = 2 : i32}], scalars = [], schema_version = 1 : i32}} : (!vrc.dispatch_desc, !vrc.buffer, !vrc.buffer) -> !vrc.event
    %d2 = vrc.dispatch_desc_create %c1 {kernel_id = 10 : i32, scalar_kinds = ["i32"]} : (i32) -> !vrc.dispatch_desc
    %e2 = vrc.dispatch %d2(%dynamic) {argument_kinds = ["buffer", "i32"], global_offset = array<i32: 0, 0, 0>, global_size = array<i32: 1, 1, 1>, kernel_id = 10 : i32, local_size = array<i32: 1, 1, 1>, resource_accesses = ["write"]} : (!vrc.dispatch_desc, !vrc.buffer) -> !vrc.event
    return
  }
}
