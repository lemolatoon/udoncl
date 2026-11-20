// RUN: %vrc-opt --vrc-assign-unity-dispatch-materials %s | %FileCheck %s
// RUN: %vrc-opt --vrc-assign-unity-dispatch-materials=max-materials=1 %s | %FileCheck %s --check-prefix=CAPPED

// Every Unity dispatch gets its own material slot in module order; the
// module lists each slot with the kernel whose shader it renders. A dispatch
// without a material slot is left alone.

// CHECK-LABEL: module attributes
// CHECK-SAME: vrc.unity_dispatch_materials = [{kernel_id = 7 : i32, slot = "__udoncl_dispatch_material_0"}, {kernel_id = 7 : i32, slot = "__udoncl_dispatch_material_1"}, {kernel_id = 9 : i32, slot = "__udoncl_dispatch_material_2"}]
// CHECK: = vrc.dispatch %
// CHECK-SAME: vrc.unity_dispatch_material_slot = "__udoncl_dispatch_material_0"
// CHECK: = vrc.dispatch %
// CHECK-SAME: vrc.unity_dispatch_material_slot = "__udoncl_dispatch_material_1"
// CHECK: = vrc.dispatch %
// CHECK-SAME: vrc.unity_dispatch_material_slot = "__udoncl_dispatch_material_2"
// CHECK: = vrc.dispatch %
// CHECK-NOT: vrc.unity_dispatch_material_slot

// With a cap, later dispatches keep their kernel's shared slot.
// CAPPED-LABEL: module attributes
// CAPPED-SAME: vrc.unity_dispatch_materials = [{kernel_id = 7 : i32, slot = "__udoncl_dispatch_material_0"}, {kernel_id = 7 : i32, slot = "__udoncl_kernel_material_7"}, {kernel_id = 9 : i32, slot = "__udoncl_kernel_material_9"}]
// CAPPED: vrc.unity_dispatch_material_slot = "__udoncl_dispatch_material_0"
// CAPPED: vrc.unity_dispatch_material_slot = "__udoncl_kernel_material_7"
// CAPPED: vrc.unity_dispatch_material_slot = "__udoncl_kernel_material_9"
module {
  func.func @sites(%input: !vrc.buffer, %output: !vrc.buffer, %n: i32) {
    %desc0 = vrc.dispatch_desc_create %n {
      kernel_id = 7 : i32, scalar_kinds = ["i32"]
    } : (i32) -> !vrc.dispatch_desc
    %event0 = vrc.dispatch %desc0(%input, %output) {
      argument_kinds = ["buffer", "buffer", "i32"],
      resource_accesses = ["read", "write"],
      global_offset = array<i32: 0, 0, 0>,
      global_size = array<i32: 1, 1, 1>,
      kernel_id = 7 : i32,
      local_size = array<i32: 1, 1, 1>,
      vrc.unity_dispatch_material_slot = "__udoncl_kernel_material_7"
    } : (!vrc.dispatch_desc, !vrc.buffer, !vrc.buffer) -> !vrc.event
    %desc1 = vrc.dispatch_desc_create %n {
      kernel_id = 7 : i32, scalar_kinds = ["i32"]
    } : (i32) -> !vrc.dispatch_desc
    %event1 = vrc.dispatch %desc1(%input, %output) {
      argument_kinds = ["buffer", "buffer", "i32"],
      resource_accesses = ["read", "write"],
      global_offset = array<i32: 0, 0, 0>,
      global_size = array<i32: 1, 1, 1>,
      kernel_id = 7 : i32,
      local_size = array<i32: 1, 1, 1>,
      vrc.unity_dispatch_material_slot = "__udoncl_kernel_material_7"
    } : (!vrc.dispatch_desc, !vrc.buffer, !vrc.buffer) -> !vrc.event
    %desc2 = vrc.dispatch_desc_create %n {
      kernel_id = 9 : i32, scalar_kinds = ["i32"]
    } : (i32) -> !vrc.dispatch_desc
    %event2 = vrc.dispatch %desc2(%input, %output) {
      argument_kinds = ["buffer", "buffer", "i32"],
      resource_accesses = ["read", "write"],
      global_offset = array<i32: 0, 0, 0>,
      global_size = array<i32: 1, 1, 1>,
      kernel_id = 9 : i32,
      local_size = array<i32: 1, 1, 1>,
      vrc.unity_dispatch_material_slot = "__udoncl_kernel_material_9"
    } : (!vrc.dispatch_desc, !vrc.buffer, !vrc.buffer) -> !vrc.event
    %desc3 = vrc.dispatch_desc_create %n {
      kernel_id = 9 : i32, scalar_kinds = ["i32"]
    } : (i32) -> !vrc.dispatch_desc
    %event3 = vrc.dispatch %desc3(%input, %output) {
      argument_kinds = ["buffer", "buffer", "i32"],
      resource_accesses = ["read", "write"],
      global_offset = array<i32: 0, 0, 0>,
      global_size = array<i32: 1, 1, 1>,
      kernel_id = 9 : i32,
      local_size = array<i32: 1, 1, 1>
    } : (!vrc.dispatch_desc, !vrc.buffer, !vrc.buffer) -> !vrc.event
    return
  }
}
