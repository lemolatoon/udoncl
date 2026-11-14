module attributes {
  vrc.static_kernel_bindings = {candidate_count = 1 : i32, schema_version = 1 : i32, snapshot_count = 1 : i32, unresolved_count = 0 : i32},
  vrc.static_dispatch_resource_materialization = {candidate_dispatch_count = 1 : i32, materialized_operand_count = 1 : i32, schema_version = 1 : i32, snapshotted_dispatch_count = 1 : i32, unresolved_dispatch_count = 0 : i32, unresolved_operand_count = 0 : i32},
  vrc.static_unity_buffer_layout_plan = {blocker_count = 0 : i32, ready = true, schema_version = 1 : i32}
} {
  func.func @ready() {
    %buffer = "vrc.assigned_buffer"() {capacity_bytes = 4 : i64, format = "rgba8_unorm", resource_id = 0 : i32, texture_id = 1 : i32} : () -> !vrc.buffer
    %descriptor = vrc.dispatch_desc_create {kernel_id = 1 : i32, scalar_kinds = []} : () -> !vrc.dispatch_desc
    %event = vrc.dispatch %descriptor(%buffer) {argument_kinds = ["buffer"], global_offset = array<i32: 0, 0, 0>, global_size = array<i32: 1, 1, 1>, kernel_id = 1 : i32, local_size = array<i32: 1, 1, 1>, resource_accesses = ["write"], vrc.static_kernel_binding_snapshot = {schema_version = 1 : i32}} : (!vrc.dispatch_desc, !vrc.buffer) -> !vrc.event
    return
  }
}

// -----

module attributes {
  vrc.static_kernel_bindings = {candidate_count = 1 : i32, schema_version = 1 : i32, snapshot_count = 1 : i32, unresolved_count = 0 : i32},
  vrc.static_dispatch_resource_materialization = {candidate_dispatch_count = 1 : i32, materialized_operand_count = 1 : i32, schema_version = 1 : i32, snapshotted_dispatch_count = 1 : i32, unresolved_dispatch_count = 0 : i32, unresolved_operand_count = 0 : i32},
  vrc.static_unity_buffer_layout_plan = {blocker_count = 0 : i32, ready = true, schema_version = 1 : i32}
} {
  func.func @resource_ready_pool_blocked() {
    %one = arith.constant 1 : i32
    %four = arith.constant 4 : i32
    %dynamic, %status = vrc.buffer_alloc %one, %four {format = "rgba8_unorm", usage = "storage"} : (i32, i32) -> (!vrc.buffer, i32)
    %released = vrc.buffer_release %dynamic : (!vrc.buffer) -> i32
    %buffer = "vrc.assigned_buffer"() {capacity_bytes = 4 : i64, format = "rgba8_unorm", resource_id = 0 : i32, texture_id = 1 : i32} : () -> !vrc.buffer
    %descriptor = vrc.dispatch_desc_create {kernel_id = 1 : i32, scalar_kinds = []} : () -> !vrc.dispatch_desc
    %event = vrc.dispatch %descriptor(%buffer) {argument_kinds = ["buffer"], global_offset = array<i32: 0, 0, 0>, global_size = array<i32: 1, 1, 1>, kernel_id = 1 : i32, local_size = array<i32: 1, 1, 1>, resource_accesses = ["write"], vrc.static_kernel_binding_snapshot = {schema_version = 1 : i32}} : (!vrc.dispatch_desc, !vrc.buffer) -> !vrc.event
    return
  }
}
