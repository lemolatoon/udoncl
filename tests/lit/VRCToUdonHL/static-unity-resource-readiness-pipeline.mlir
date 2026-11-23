// RUN: %vrc-opt --vrc-static-unity-resource-readiness-pipeline %s | %FileCheck %s

// The pipeline replaces the proven allocation operand of a dispatch with its
// assigned Unity Texture and assesses the static Unity resource lowering.

// CHECK: vrc.static_unity_resource_lowering_readiness = {
// CHECK-SAME: pool_replacement_ready = true
// CHECK-SAME: resource_operands_ready = true
// CHECK-LABEL: func.func @run
// CHECK: %[[ASSIGNED:.*]] = "vrc.assigned_buffer"() <{capacity_bytes = 32 : i64, format = "rgba8_unorm", resource_id = 0 : i32, texture_id = 1 : i32}>
// CHECK: vrc.dispatch %{{.*}}(%[[ASSIGNED]])
module attributes {
  vrc.static_device_resource_inventory = {schema_version = 1 : i32, resources = [
    {capacity_bytes = 32 : i64, format = "rgba8_unorm", persistent = true, resource_id = 0 : i32, resource_kind = "allocation"}
  ]},
  vrc.unity_texture_assignments = {schema_version = 1 : i32, assignments = [
    {capacity_bytes = 32 : i64, format = "rgba8_unorm", persistent = true, resource_id = 0 : i32, resource_kind = "allocation", texture_id = 1 : i32}
  ]},
  vrc.static_kernel_bindings = {candidate_count = 1 : i32, schema_version = 1 : i32, snapshot_count = 1 : i32, unresolved_count = 0 : i32},
  vrc.static_unity_buffer_layout_plan = {blocker_count = 0 : i32, ready = true, schema_version = 1 : i32}
} {
  func.func @run(%buffer: !vrc.buffer) {
    %descriptor = vrc.dispatch_desc_create {kernel_id = 1 : i32, scalar_kinds = []} : () -> !vrc.dispatch_desc
    %event = vrc.dispatch %descriptor(%buffer) {argument_kinds = ["buffer"], global_offset = array<i32: 0, 0, 0>, global_size = array<i32: 1, 1, 1>, kernel_id = 1 : i32, local_size = array<i32: 1, 1, 1>, resource_accesses = ["write"], vrc.static_kernel_binding_snapshot = {kernel_id = 1 : i32, resources = [{access = "write", kind = "allocation", operand_index = 0 : i32, resource_id = 0 : i32}], scalars = [], schema_version = 1 : i32}} : (!vrc.dispatch_desc, !vrc.buffer) -> !vrc.event
    return
  }
}
