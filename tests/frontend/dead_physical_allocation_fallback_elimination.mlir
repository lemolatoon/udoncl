module attributes {
  vrc.static_kernel_bindings = {candidate_count = 1 : i32, schema_version = 1 : i32, snapshot_count = 1 : i32, unresolved_count = 0 : i32}
} {
  func.func private @allocator_sink(%logical: i32, %bytes: i32) -> (!vrc.buffer, i32) {
    %buffer, %status = vrc.buffer_alloc %logical, %bytes {format = "rgba8_unorm", usage = "storage"} : (i32, i32) -> (!vrc.buffer, i32)
    return %buffer, %status : !vrc.buffer, i32
  }

  func.func private @allocator_wrapper(%logical: i32, %bytes: i32) -> (!vrc.buffer, i32) {
    %buffer, %status = call @allocator_sink(%logical, %bytes) : (i32, i32) -> (!vrc.buffer, i32)
    return %buffer, %status : !vrc.buffer, i32
  }

  func.func private @dead_dispatch() {
    %buffer = "vrc.assigned_buffer"() {capacity_bytes = 16 : i64, format = "rgba8_unorm", resource_id = 0 : i32, texture_id = 1 : i32} : () -> !vrc.buffer
    %descriptor = "vrc.dispatch_desc_create"() {kernel_id = 7 : i32, scalar_kinds = []} : () -> !vrc.dispatch_desc
    %event = vrc.dispatch %descriptor(%buffer) {argument_kinds = ["buffer"], global_offset = array<i32: 0, 0, 0>, global_size = array<i32: 1, 1, 1>, kernel_id = 7 : i32, local_size = array<i32: 1, 1, 1>, resource_accesses = ["write"], vrc.static_kernel_binding_snapshot = {kernel_id = 7 : i32, resources = [{access = "write", kind = "allocation", operand_index = 0 : i32, resource_id = 0 : i32}], scalars = [], schema_version = 1 : i32}} : (!vrc.dispatch_desc, !vrc.buffer) -> !vrc.event
    return
  }

  func.func @root() {
    return
  }
}
