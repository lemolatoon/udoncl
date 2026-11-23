// RUN: %vrc-opt --vrc-materialize-static-dispatch-resources %s | %FileCheck %s

// A snapshotted dispatch operand that is already a compiler-assigned buffer
// (through the casts between the buffer and its OpenCL token) needs no
// Texture of the inventory: it takes the assigned buffer itself and counts
// as materialized.

// CHECK-LABEL: module attributes
// CHECK-SAME: vrc.static_dispatch_resource_materialization = {candidate_dispatch_count = 1 : i32, materialized_operand_count = 1 : i32, schema_version = 1 : i32, snapshotted_dispatch_count = 1 : i32, unresolved_dispatch_count = 0 : i32, unresolved_operand_count = 0 : i32}
// CHECK: %[[ASSIGNED:.*]] = "vrc.assigned_buffer"() <{capacity_bytes = 4 : i64, format = "rgba8_unorm", resource_id = 3 : i32, texture_id = 1 : i32}>
// CHECK: vrc.dispatch %{{.*}}(%[[ASSIGNED]])
module attributes {
  vrc.static_device_resource_inventory = {schema_version = 1 : i32, resources = []},
  vrc.unity_texture_assignments = {schema_version = 1 : i32, assignments = []}
} {
  func.func @run() {
    %c1 = arith.constant 1 : i32
    %assigned = "vrc.assigned_buffer"() <{capacity_bytes = 4 : i64, format = "rgba8_unorm", resource_id = 3 : i32, texture_id = 1 : i32}> : () -> !vrc.buffer
    %token = builtin.unrealized_conversion_cast %assigned : !vrc.buffer to i32
    %buffer = builtin.unrealized_conversion_cast %token : i32 to !vrc.buffer
    %descriptor = vrc.dispatch_desc_create %c1 {kernel_id = 8 : i32, scalar_kinds = ["i32"]} : (i32) -> !vrc.dispatch_desc
    %event = vrc.dispatch %descriptor(%buffer) {argument_kinds = ["buffer", "i32"], global_offset = array<i32: 0, 0, 0>, global_size = array<i32: 1, 1, 1>, kernel_id = 8 : i32, local_size = array<i32: 1, 1, 1>, resource_accesses = ["write"], vrc.static_kernel_binding_snapshot = {kernel_id = 8 : i32, resources = [{access = "write", kind = "allocation", operand_index = 0 : i32, resource_id = 3 : i32}], schema_version = 1 : i32}} : (!vrc.dispatch_desc, !vrc.buffer) -> !vrc.event
    return
  }
}
