// RUN: %vrc-opt --split-input-file --vrc-plan-assigned-unity-buffer-layout %s | %FileCheck %s

// The textures a Host names with vrc.assigned_buffer are its Unity buffers.

// CHECK-LABEL: module attributes
// CHECK-SAME: vrc.static_unity_buffer_layout_plan = {buffers = [{buffer_id = 1 : i32, capacity_bytes = 4 : i64, format = "rgba8_unorm", heap_slot = "__udoncl_buffer_1", persistent = true, public = false, resource_id = 0 : i32, texture_id = 1 : i32}, {buffer_id = 2 : i32, capacity_bytes = 8 : i64, format = "rgba8_unorm", heap_slot = "__udoncl_buffer_2", persistent = true, public = false, resource_id = 1 : i32, texture_id = 2 : i32}], ready = true, schema_version = 1 : i32}
// CHECK-SAME: vrc.static_unity_resource_lowering_readiness = {applied = true, pool_replacement_ready = true, schema_version = 1 : i32}
module attributes {vrc.static_unity_resource_lowering_readiness = {applied = false, pool_replacement_ready = false, schema_version = 1 : i32}} {
  func.func @copy() {
    %source = "vrc.assigned_buffer"() <{capacity_bytes = 4 : i64, format = "rgba8_unorm", resource_id = 0 : i32, texture_id = 1 : i32}> : () -> !vrc.buffer
    %destination = "vrc.assigned_buffer"() <{capacity_bytes = 8 : i64, format = "rgba8_unorm", resource_id = 1 : i32, texture_id = 2 : i32}> : () -> !vrc.buffer
    %again = "vrc.assigned_buffer"() <{capacity_bytes = 4 : i64, format = "rgba8_unorm", resource_id = 0 : i32, texture_id = 1 : i32}> : () -> !vrc.buffer
    return
  }
}

// -----

// A Host that still allocates Device buffers, or whose static Device
// resource lowering is applied, is left alone.

// CHECK-LABEL: module {
// CHECK-NOT: vrc.static_unity_buffer_layout_plan
module {
  func.func private @allocate(i32, i32) -> (!vrc.buffer, i32) attributes {vrc.intrinsic = "buffer_alloc"}
  func.func @run() {
    %id = arith.constant 1 : i32
    %bytes = arith.constant 4 : i32
    %buffer, %status = func.call @allocate(%id, %bytes) : (i32, i32) -> (!vrc.buffer, i32)
    %assigned = "vrc.assigned_buffer"() <{capacity_bytes = 4 : i64, format = "rgba8_unorm", resource_id = 0 : i32, texture_id = 1 : i32}> : () -> !vrc.buffer
    return
  }
}

// -----

// CHECK-LABEL: module attributes {vrc.static_unity_resource_lowering_readiness = {applied = true, pool_replacement_ready = true, schema_version = 1 : i32}} {
// CHECK-NOT: vrc.static_unity_buffer_layout_plan
module attributes {vrc.static_unity_resource_lowering_readiness = {applied = true, pool_replacement_ready = true, schema_version = 1 : i32}} {
  func.func @run() {
    %assigned = "vrc.assigned_buffer"() <{capacity_bytes = 4 : i64, format = "rgba8_unorm", resource_id = 0 : i32, texture_id = 1 : i32}> : () -> !vrc.buffer
    return
  }
}
