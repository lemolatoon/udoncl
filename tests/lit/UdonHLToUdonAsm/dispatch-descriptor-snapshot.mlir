// RUN: %vrc-opt --vrc-lower-udonhl-to-udonasm %s | %FileCheck %s

// A descriptor whose dispatch does not directly follow it keeps its scalar
// snapshot: an operation in between may reuse the scalars' slots.

module attributes {
  vrc.host_stage = "udonhl",
  vrc.unity_buffer_pool = [
    {id = 1 : i32, logical_bytes = 4 : i32, heap_slot = "__udoncl_buffer_1"},
    {id = 2 : i32, logical_bytes = 4 : i32, heap_slot = "__udoncl_buffer_2"}
  ]
} {
  // CHECK-DAG: name = "[[SNAPSHOT:[^"]*descriptor_0_0_SystemInt32]]"

  // CHECK: udonasm.push "__0_arg0__param"
  // CHECK-NEXT: udonasm.push "[[SNAPSHOT]]"
  // CHECK-NEXT: udonasm.copy
  // CHECK: udonasm.push "__udoncl_material_arg_2"
  // CHECK-NEXT: udonasm.push "[[SNAPSHOT]]"
  // CHECK-NEXT: udonasm.extern "UnityEngineMaterial.__SetInteger__SystemString_SystemInt32__SystemVoid"
  func.func @separated(%count: i32) -> i32 attributes {vrc.export_id = 1 : i32} {
    %input_id = arith.constant 1 : i32
    %output_id = arith.constant 2 : i32
    %bytes = arith.constant 4 : i32
    %input, %input_status = vrc.buffer_alloc %input_id, %bytes {
      format = "rgba8_unorm", usage = "storage"
    } : (i32, i32) -> (!vrc.buffer, i32)
    %output, %output_status = vrc.buffer_alloc %output_id, %bytes {
      format = "rgba8_unorm", usage = "storage"
    } : (i32, i32) -> (!vrc.buffer, i32)
    %desc = vrc.dispatch_desc_create %count {
      kernel_id = 1 : i32, scalar_kinds = ["i32"]
    } : (i32) -> !vrc.dispatch_desc
    %other = arith.addi %bytes, %bytes : i32
    %event = vrc.dispatch %desc(%input, %output) {
      argument_kinds = ["buffer", "buffer", "i32"],
      resource_accesses = ["read", "write"],
      global_offset = array<i32: 0, 0, 0>,
      global_size = array<i32: 1, 1, 1>,
      kernel_id = 1 : i32,
      local_size = array<i32: 1, 1, 1>,
      vrc.unity_dispatch_material_slot = "__udoncl_kernel_material_1",
      vrc.unity_output_resource_index = 1 : i32,
      vrc.unity_resource_heap_slots = ["", ""]
    } : (!vrc.dispatch_desc, !vrc.buffer, !vrc.buffer) -> !vrc.event
    return %other : i32
  }
}
