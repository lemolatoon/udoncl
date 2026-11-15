module {
  func.func @run(%dynamic: !vrc.buffer) {
    %static0 = "vrc.static_buffer"() <{byte_size = 16 : i32, resource_id = 4 : i32}> : () -> !vrc.buffer
    %static1 = "vrc.static_buffer"() <{byte_size = 32 : i32, resource_id = 9 : i32}> : () -> !vrc.buffer
    %descriptor0 = vrc.dispatch_desc_create {kernel_id = 7 : i32, scalar_kinds = []} : () -> !vrc.dispatch_desc
    %event0 = vrc.dispatch %descriptor0(%static0, %dynamic, %static1) {argument_kinds = ["buffer", "buffer", "buffer"], global_offset = array<i32: 0, 0, 0>, global_size = array<i32: 1, 1, 1>, kernel_id = 7 : i32, local_size = array<i32: 1, 1, 1>, resource_accesses = ["read", "write", "read"], vrc.static_kernel_binding_snapshot = {kernel_id = 7 : i32, resources = [{access = "read", kind = "static_tensor", operand_index = 2 : i32, resource_id = 9 : i32}, {access = "write", kind = "allocation", operand_index = 1 : i32, resource_id = 2 : i32}, {access = "read", kind = "static_tensor", operand_index = 0 : i32, resource_id = 4 : i32}], scalars = [], schema_version = 1 : i32}} : (!vrc.dispatch_desc, !vrc.buffer, !vrc.buffer, !vrc.buffer) -> !vrc.event
    %descriptor1 = vrc.dispatch_desc_create {kernel_id = 8 : i32, scalar_kinds = []} : () -> !vrc.dispatch_desc
    %event1 = vrc.dispatch %descriptor1(%dynamic) {argument_kinds = ["buffer"], global_offset = array<i32: 0, 0, 0>, global_size = array<i32: 1, 1, 1>, kernel_id = 8 : i32, local_size = array<i32: 1, 1, 1>, resource_accesses = ["write"], vrc.static_kernel_binding_snapshot = {kernel_id = 8 : i32, resources = [{access = "write", kind = "allocation", operand_index = 0 : i32, resource_id = 2 : i32}], scalars = [], schema_version = 1 : i32}} : (!vrc.dispatch_desc, !vrc.buffer) -> !vrc.event
    return
  }
}
