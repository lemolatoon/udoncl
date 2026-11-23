// RUN: %vrc-opt --vrc-static-unity-resource-readiness-pipeline --split-input-file %s | %FileCheck %s

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

// -----

// A transfer call proven to write one allocation calls a specialization of
// the runtime transfer that uploads to its assigned Texture, so the
// transfers no longer block the static Unity resource lowering.

// CHECK: vrc.static_unity_resource_lowering_readiness = {
// CHECK-SAME: pool_replacement_ready = true
// CHECK-SAME: transfer_call_count = 0 : i32, transfer_ready = true
// CHECK-SAME: typed_upload_count = 1 : i32
// CHECK-SAME: unresolved_typed_transfer_operand_count = 0 : i32
// CHECK-NOT: func.func @clEnqueueWriteBuffer(
// CHECK-LABEL: func.func private @__vrc_static_clEnqueueWriteBuffer_1_0(
// CHECK: %[[ASSIGNED:.*]] = "vrc.assigned_buffer"()
// CHECK: vrc.buffer_upload %[[ASSIGNED]]
// CHECK-LABEL: func.func @write
// CHECK: call @__vrc_static_clEnqueueWriteBuffer_1_0(
module attributes {
  vrc.static_device_resource_inventory = {schema_version = 1 : i32, resources = [
    {capacity_bytes = 32 : i64, format = "rgba8_unorm", persistent = true, resource_id = 0 : i32, resource_kind = "allocation"}
  ]},
  vrc.unity_texture_assignments = {schema_version = 1 : i32, assignments = [
    {capacity_bytes = 32 : i64, format = "rgba8_unorm", persistent = true, resource_id = 0 : i32, resource_kind = "allocation", texture_id = 1 : i32}
  ]},
  vrc.static_transfer_bindings = {candidate_count = 1 : i32, requires_context_specialization_count = 0 : i32, resolved_count = 1 : i32, schema_version = 1 : i32, unresolved_count = 0 : i32},
  vrc.static_unity_buffer_layout_plan = {blocker_count = 0 : i32, ready = true, schema_version = 1 : i32}
} {
  func.func private @udoncl_mem_physical_token(!udonhl.ptr<0>) -> i32
  func.func @clEnqueueWriteBuffer(%queue: !udonhl.ptr<0>, %buffer: !udonhl.ptr<0>, %offset: i32, %bytes: i32, %source: i32) -> i32 {
    %token = func.call @udoncl_mem_physical_token(%buffer) : (!udonhl.ptr<0>) -> i32
    %physical = builtin.unrealized_conversion_cast %token : i32 to !vrc.buffer
    %event = vrc.buffer_upload %physical, %offset, %bytes, %source : (!vrc.buffer, i32, i32, i32) -> !vrc.event
    %status = builtin.unrealized_conversion_cast %event : !vrc.event to i32
    return %status : i32
  }
  func.func @write(%queue: !udonhl.ptr<0>, %buffer: !udonhl.ptr<0>, %source: i32) -> i32 attributes {udoncl.export} {
    %zero = arith.constant 0 : i32
    %bytes = arith.constant 32 : i32
    %status = func.call @clEnqueueWriteBuffer(%queue, %buffer, %zero, %bytes, %source) {vrc.static_transfer_binding_snapshot = {cases = [{call_path = [], resources = [{access = "write", operand_index = 1 : i32, resource_id = 0 : i32, resource_kind = "allocation"}], root = @write}], requires_context_specialization = false, schema_version = 1 : i32, transfer_kind = "write"}} : (!udonhl.ptr<0>, !udonhl.ptr<0>, i32, i32, i32) -> i32
    return %status : i32
  }
}
