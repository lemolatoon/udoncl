// RUN: %vrc-opt --vrc-assign-unity-dispatch-materials=max-materials=1 --verify-diagnostics %s

// Beyond the cap a dispatch keeps its slot, which must not name the material
// of another kernel.
module {
  func.func @conflict(%input: !vrc.buffer, %output: !vrc.buffer) {
    %desc0 = vrc.dispatch_desc_create {
      kernel_id = 7 : i32, scalar_kinds = []
    } : () -> !vrc.dispatch_desc
    %event0 = vrc.dispatch %desc0(%input, %output) {
      argument_kinds = ["buffer", "buffer"],
      resource_accesses = ["read", "write"],
      global_offset = array<i32: 0, 0, 0>,
      global_size = array<i32: 1, 1, 1>,
      kernel_id = 7 : i32,
      local_size = array<i32: 1, 1, 1>,
      vrc.unity_dispatch_material_slot = "__udoncl_kernel_material_7"
    } : (!vrc.dispatch_desc, !vrc.buffer, !vrc.buffer) -> !vrc.event
    %desc1 = vrc.dispatch_desc_create {
      kernel_id = 7 : i32, scalar_kinds = []
    } : () -> !vrc.dispatch_desc
    %event1 = vrc.dispatch %desc1(%input, %output) {
      argument_kinds = ["buffer", "buffer"],
      resource_accesses = ["read", "write"],
      global_offset = array<i32: 0, 0, 0>,
      global_size = array<i32: 1, 1, 1>,
      kernel_id = 7 : i32,
      local_size = array<i32: 1, 1, 1>,
      vrc.unity_dispatch_material_slot = "__udoncl_kernel_material_7"
    } : (!vrc.dispatch_desc, !vrc.buffer, !vrc.buffer) -> !vrc.event
    %desc2 = vrc.dispatch_desc_create {
      kernel_id = 9 : i32, scalar_kinds = []
    } : () -> !vrc.dispatch_desc
    // expected-error @+1 {{material slot __udoncl_kernel_material_7 is shared by kernels 7 and 9}}
    %event2 = vrc.dispatch %desc2(%input, %output) {
      argument_kinds = ["buffer", "buffer"],
      resource_accesses = ["read", "write"],
      global_offset = array<i32: 0, 0, 0>,
      global_size = array<i32: 1, 1, 1>,
      kernel_id = 9 : i32,
      local_size = array<i32: 1, 1, 1>,
      vrc.unity_dispatch_material_slot = "__udoncl_kernel_material_7"
    } : (!vrc.dispatch_desc, !vrc.buffer, !vrc.buffer) -> !vrc.event
    return
  }
}
