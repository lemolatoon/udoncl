// RUN: %vrc-opt --split-input-file --vrc-lower-udonhl-to-udonasm %s | %FileCheck %s

// A kernel material keeps its properties between Blits, and only the
// dispatches through it write them. The first dispatch through the material
// binds every property the material keeps and sets its bound flag:
//   * an argument every dispatch binds to the same fixed value (a constant, a
//     static tensor or a fixed buffer pool entry) is bound only then;
//   * an i32 scalar, a token-selected buffer and a token-selected render
//     target are remembered in per-material slots; a later dispatch compares
//     the value with the remembered one and rebinds only on a change;
//   * anything else (a run-time f32) is set by every dispatch.

module attributes {
  vrc.host_stage = "udonhl",
  vrc.unity_buffer_pool = [
    {id = 1 : i32, logical_bytes = 4 : i32, heap_slot = "__udoncl_buffer_1"}
  ]
} {
  // CHECK-LABEL: module
  // CHECK-DAG: {initial = "null", name = "__udoncl_bound_kernel_material_1", type = "SystemBoolean"}
  // CHECK-DAG: {initial = "0", name = "__udoncl_arg_3_kernel_material_1", type = "SystemInt32"}
  // CHECK-DAG: {initial = "null", name = "__udoncl_target_kernel_material_1", type = "UnityEngineRenderTexture"}
  // CHECK-DAG: {initial = "0", name = "__udoncl_target_token_kernel_material_1", type = "SystemInt32"}
  // CHECK-DAG: {initial = "41", name = "[[LITERAL:__vrc_constant_[0-9]+]]", type = "SystemInt32"}
  // CHECK-DAG: {initial = "1.5", name = "[[FLOAT:__vrc_constant_[0-9]+]]", type = "SystemSingle"}
  // CHECK-DAG: {initial = "7", name = "[[INDEX:__vrc_constant_[0-9]+]]", type = "SystemInt32"}

  // Steady state: the run-time count and the render target token are
  // compared with the remembered values.
  // CHECK: udonasm.push "__udoncl_bound_kernel_material_1"
  // CHECK-NEXT: udonasm.jump_if_false
  // CHECK-NEXT: udonasm.push "__0_arg0__param"
  // CHECK-NEXT: udonasm.push "__udoncl_arg_3_kernel_material_1"
  // CHECK-NEXT: udonasm.push "__udoncl_material_value_changed"
  // CHECK-NEXT: udonasm.extern "SystemInt32.__op_Inequality__SystemInt32_SystemInt32__SystemBoolean"
  // CHECK-NEXT: udonasm.push "__udoncl_material_value_changed"
  // CHECK-NEXT: udonasm.jump_if_false
  // CHECK-NEXT: udonasm.push "__udoncl_kernel_material_1"
  // CHECK-NEXT: udonasm.push "__udoncl_material_arg_3"
  // CHECK-NEXT: udonasm.push "__0_arg0__param"
  // CHECK-NEXT: udonasm.extern "UnityEngineMaterial.__SetInteger__SystemString_SystemInt32__SystemVoid"
  // CHECK-NEXT: udonasm.push "__0_arg0__param"
  // CHECK-NEXT: udonasm.push "__udoncl_arg_3_kernel_material_1"
  // CHECK-NEXT: udonasm.copy
  // CHECK-NEXT: udonasm.push "[[TOKEN:[^"]+]]"
  // CHECK-NEXT: udonasm.push "__udoncl_target_token_kernel_material_1"
  // CHECK-NEXT: udonasm.push "__udoncl_material_value_changed"
  // CHECK-NEXT: udonasm.extern "SystemInt32.__op_Inequality__SystemInt32_SystemInt32__SystemBoolean"
  // CHECK: udonasm.push "__udoncl_target_kernel_material_1"
  // CHECK-NEXT: udonasm.copy
  // CHECK-NEXT: udonasm.push "[[TOKEN]]"
  // CHECK-NEXT: udonasm.push "__udoncl_target_token_kernel_material_1"
  // CHECK-NEXT: udonasm.copy
  // CHECK-NEXT: udonasm.jump
  // First dispatch: the render target and its token, the static tensor, 41
  // and 1.5, and the count with its remembered value.
  // CHECK: udonasm.push "__udoncl_target_kernel_material_1"
  // CHECK-NEXT: udonasm.copy
  // CHECK-NEXT: udonasm.push "[[TOKEN]]"
  // CHECK-NEXT: udonasm.push "__udoncl_target_token_kernel_material_1"
  // CHECK-NEXT: udonasm.copy
  // CHECK-NEXT: udonasm.push "__udoncl_static_tensors"
  // CHECK-NEXT: udonasm.push "[[INDEX]]"
  // CHECK-NEXT: udonasm.push "[[STATIC:[^"]+]]"
  // CHECK-NEXT: udonasm.extern "UnityEngineTexture2DArray.__Get__SystemInt32__UnityEngineTexture2D"
  // CHECK-NEXT: udonasm.push "__udoncl_kernel_material_1"
  // CHECK-NEXT: udonasm.push "__udoncl_material_input_0"
  // CHECK-NEXT: udonasm.push "[[STATIC]]"
  // CHECK-NEXT: udonasm.extern "UnityEngineMaterial.__SetTexture__SystemString_UnityEngineTexture__SystemVoid"
  // CHECK-NEXT: udonasm.push "__udoncl_kernel_material_1"
  // CHECK-NEXT: udonasm.push "__udoncl_material_arg_2"
  // CHECK-NEXT: udonasm.push "[[LITERAL]]"
  // CHECK-NEXT: udonasm.extern "UnityEngineMaterial.__SetInteger__SystemString_SystemInt32__SystemVoid"
  // CHECK-NEXT: udonasm.push "__udoncl_kernel_material_1"
  // CHECK-NEXT: udonasm.push "__udoncl_material_arg_3"
  // CHECK-NEXT: udonasm.push "__0_arg0__param"
  // CHECK-NEXT: udonasm.extern "UnityEngineMaterial.__SetInteger__SystemString_SystemInt32__SystemVoid"
  // CHECK-NEXT: udonasm.push "__0_arg0__param"
  // CHECK-NEXT: udonasm.push "__udoncl_arg_3_kernel_material_1"
  // CHECK-NEXT: udonasm.copy
  // CHECK-NEXT: udonasm.push "__udoncl_kernel_material_1"
  // CHECK-NEXT: udonasm.push "__udoncl_material_arg_4"
  // CHECK-NEXT: udonasm.push "[[FLOAT]]"
  // CHECK-NEXT: udonasm.extern "UnityEngineMaterial.__SetFloat__SystemString_SystemSingle__SystemVoid"
  // CHECK-NEXT: udonasm.push "[[ONE:[^"]+]]"
  // CHECK-NEXT: udonasm.push "[[ONE]]"
  // CHECK-NEXT: udonasm.push "__udoncl_bound_kernel_material_1"
  // CHECK-NEXT: udonasm.extern "SystemInt32.__op_Equality__SystemInt32_SystemInt32__SystemBoolean"
  // CHECK-NEXT: udonasm.push "__udoncl_null_texture"
  // CHECK-NEXT: udonasm.push "__udoncl_target_kernel_material_1"
  // CHECK-NEXT: udonasm.push "__udoncl_kernel_material_1"
  // CHECK-NEXT: udonasm.extern "VRCSDKBaseVRCGraphics.__Blit__
  // The second dispatch through the material shares its state.
  // CHECK: udonasm.push "__udoncl_bound_kernel_material_1"
  // CHECK-NEXT: udonasm.jump_if_false
  // CHECK-NEXT: udonasm.push "__0_arg0__param"
  // CHECK-NEXT: udonasm.push "__udoncl_arg_3_kernel_material_1"
  // CHECK: udonasm.extern "VRCSDKBaseVRCGraphics.__Blit__
  func.func @uniform(%count: i32) attributes {vrc.export_id = 1 : i32} {
    %output_id = arith.constant 1 : i32
    %bytes = arith.constant 4 : i32
    %literal = arith.constant 41 : i32
    %scale = arith.constant 1.5 : f32
    %weights = "vrc.static_buffer"() <{byte_size = 16 : i32, resource_id = 7 : i32}> : () -> !vrc.buffer
    %output, %output_status = vrc.buffer_alloc %output_id, %bytes {
      format = "rgba8_unorm", usage = "storage"
    } : (i32, i32) -> (!vrc.buffer, i32)
    %desc0 = vrc.dispatch_desc_create %literal, %count, %scale {
      kernel_id = 1 : i32, scalar_kinds = ["i32", "i32", "f32"]
    } : (i32, i32, f32) -> !vrc.dispatch_desc
    %event0 = vrc.dispatch %desc0(%weights, %output) {
      argument_kinds = ["buffer", "buffer", "i32", "i32", "f32"],
      resource_accesses = ["read", "write"],
      global_offset = array<i32: 0, 0, 0>,
      global_size = array<i32: 1, 1, 1>,
      kernel_id = 1 : i32,
      local_size = array<i32: 1, 1, 1>,
      vrc.unity_dispatch_material_slot = "__udoncl_kernel_material_1",
      vrc.unity_output_resource_index = 1 : i32,
      vrc.unity_resource_heap_slots = ["", ""]
    } : (!vrc.dispatch_desc, !vrc.buffer, !vrc.buffer) -> !vrc.event
    %desc1 = vrc.dispatch_desc_create %literal, %count, %scale {
      kernel_id = 1 : i32, scalar_kinds = ["i32", "i32", "f32"]
    } : (i32, i32, f32) -> !vrc.dispatch_desc
    %event1 = vrc.dispatch %desc1(%weights, %output) {
      argument_kinds = ["buffer", "buffer", "i32", "i32", "f32"],
      resource_accesses = ["read", "write"],
      global_offset = array<i32: 0, 0, 0>,
      global_size = array<i32: 1, 1, 1>,
      kernel_id = 1 : i32,
      local_size = array<i32: 1, 1, 1>,
      vrc.unity_dispatch_material_slot = "__udoncl_kernel_material_1",
      vrc.unity_output_resource_index = 1 : i32,
      vrc.unity_resource_heap_slots = ["", ""]
    } : (!vrc.dispatch_desc, !vrc.buffer, !vrc.buffer) -> !vrc.event
    return
  }
}

// -----

// Two dispatches through one material that bind different static tensors to
// an argument set it every time; different constants of an i32 argument are
// remembered like a run-time value; an argument they agree on is still bound
// once.

module attributes {
  vrc.host_stage = "udonhl",
  vrc.unity_buffer_pool = [
    {id = 1 : i32, logical_bytes = 4 : i32, heap_slot = "__udoncl_buffer_1"}
  ]
} {
  // CHECK-LABEL: module
  // CHECK-DAG: {initial = "41", name = "[[FIRST:__vrc_constant_[0-9]+]]", type = "SystemInt32"}
  // CHECK-DAG: {initial = "42", name = "[[SECOND:__vrc_constant_[0-9]+]]", type = "SystemInt32"}
  // CHECK-DAG: {initial = "9", name = "[[SHARED:__vrc_constant_[0-9]+]]", type = "SystemInt32"}

  // CHECK: udonasm.push "__udoncl_bound_kernel_material_1"
  // CHECK-NEXT: udonasm.jump_if_false
  // CHECK-NEXT: udonasm.push "[[FIRST]]"
  // CHECK-NEXT: udonasm.push "__udoncl_arg_2_kernel_material_1"
  // CHECK-NOT: udonasm.push "__udoncl_material_input_0"
  // CHECK: udonasm.jump
  // CHECK: udonasm.push "__udoncl_material_arg_2"
  // CHECK-NEXT: udonasm.push "[[FIRST]]"
  // CHECK: udonasm.push "__udoncl_material_arg_3"
  // CHECK-NEXT: udonasm.push "[[SHARED]]"
  // CHECK: udonasm.extern "SystemInt32.__op_Equality__SystemInt32_SystemInt32__SystemBoolean"
  // CHECK-NEXT: udonasm.push "__udoncl_static_tensors"
  // CHECK: udonasm.push "__udoncl_material_input_0"
  // CHECK: udonasm.extern "VRCSDKBaseVRCGraphics.__Blit__
  // CHECK: udonasm.push "__udoncl_bound_kernel_material_1"
  // CHECK-NEXT: udonasm.jump_if_false
  // CHECK-NEXT: udonasm.push "[[SECOND]]"
  // CHECK-NEXT: udonasm.push "__udoncl_arg_2_kernel_material_1"
  // CHECK: udonasm.push "__udoncl_material_arg_3"
  // CHECK-NEXT: udonasm.push "[[SHARED]]"
  // CHECK: udonasm.push "__udoncl_material_input_0"
  // CHECK: udonasm.extern "VRCSDKBaseVRCGraphics.__Blit__
  func.func @divergent() attributes {vrc.export_id = 1 : i32} {
    %output_id = arith.constant 1 : i32
    %bytes = arith.constant 4 : i32
    %first = arith.constant 41 : i32
    %second = arith.constant 42 : i32
    %shared = arith.constant 9 : i32
    %weights0 = "vrc.static_buffer"() <{byte_size = 16 : i32, resource_id = 3 : i32}> : () -> !vrc.buffer
    %weights1 = "vrc.static_buffer"() <{byte_size = 16 : i32, resource_id = 4 : i32}> : () -> !vrc.buffer
    %output, %output_status = vrc.buffer_alloc %output_id, %bytes {
      format = "rgba8_unorm", usage = "storage"
    } : (i32, i32) -> (!vrc.buffer, i32)
    %desc0 = vrc.dispatch_desc_create %first, %shared {
      kernel_id = 1 : i32, scalar_kinds = ["i32", "i32"]
    } : (i32, i32) -> !vrc.dispatch_desc
    %event0 = vrc.dispatch %desc0(%weights0, %output) {
      argument_kinds = ["buffer", "buffer", "i32", "i32"],
      resource_accesses = ["read", "write"],
      global_offset = array<i32: 0, 0, 0>,
      global_size = array<i32: 1, 1, 1>,
      kernel_id = 1 : i32,
      local_size = array<i32: 1, 1, 1>,
      vrc.unity_dispatch_material_slot = "__udoncl_kernel_material_1",
      vrc.unity_output_resource_index = 1 : i32,
      vrc.unity_resource_heap_slots = ["", ""]
    } : (!vrc.dispatch_desc, !vrc.buffer, !vrc.buffer) -> !vrc.event
    %desc1 = vrc.dispatch_desc_create %second, %shared {
      kernel_id = 1 : i32, scalar_kinds = ["i32", "i32"]
    } : (i32, i32) -> !vrc.dispatch_desc
    %event1 = vrc.dispatch %desc1(%weights1, %output) {
      argument_kinds = ["buffer", "buffer", "i32", "i32"],
      resource_accesses = ["read", "write"],
      global_offset = array<i32: 0, 0, 0>,
      global_size = array<i32: 1, 1, 1>,
      kernel_id = 1 : i32,
      local_size = array<i32: 1, 1, 1>,
      vrc.unity_dispatch_material_slot = "__udoncl_kernel_material_1",
      vrc.unity_output_resource_index = 1 : i32,
      vrc.unity_resource_heap_slots = ["", ""]
    } : (!vrc.dispatch_desc, !vrc.buffer, !vrc.buffer) -> !vrc.event
    return
  }
}

// -----

// A run-time i32 scalar and a token-selected render target are remembered.

module attributes {
  vrc.host_stage = "udonhl",
  vrc.unity_buffer_pool = [
    {id = 1 : i32, logical_bytes = 4 : i32, heap_slot = "__udoncl_buffer_1"}
  ]
} {
  // CHECK-LABEL: module
  // CHECK: udonasm.push "__udoncl_bound_kernel_material_1"
  // CHECK-NEXT: udonasm.jump_if_false
  // CHECK-NEXT: udonasm.push "__0_arg0__param"
  // CHECK-NEXT: udonasm.push "__udoncl_arg_1_kernel_material_1"
  func.func @runtime(%count: i32) attributes {vrc.export_id = 1 : i32} {
    %output_id = arith.constant 1 : i32
    %bytes = arith.constant 4 : i32
    %output, %output_status = vrc.buffer_alloc %output_id, %bytes {
      format = "rgba8_unorm", usage = "storage"
    } : (i32, i32) -> (!vrc.buffer, i32)
    %desc = vrc.dispatch_desc_create %count {
      kernel_id = 1 : i32, scalar_kinds = ["i32"]
    } : (i32) -> !vrc.dispatch_desc
    %event = vrc.dispatch %desc(%output) {
      argument_kinds = ["buffer", "i32"],
      resource_accesses = ["write"],
      global_offset = array<i32: 0, 0, 0>,
      global_size = array<i32: 1, 1, 1>,
      kernel_id = 1 : i32,
      local_size = array<i32: 1, 1, 1>,
      vrc.unity_dispatch_material_slot = "__udoncl_kernel_material_1",
      vrc.unity_output_resource_index = 0 : i32,
      vrc.unity_resource_heap_slots = [""]
    } : (!vrc.dispatch_desc, !vrc.buffer) -> !vrc.event
    return
  }
}

// -----

// A render target that is a fixed pool entry for one dispatch and a token for
// the other, and a run-time f32: nothing is kept, no bound flag exists and
// every dispatch binds everything.

module attributes {
  vrc.host_stage = "udonhl",
  vrc.unity_buffer_pool = [
    {id = 1 : i32, logical_bytes = 4 : i32, heap_slot = "__udoncl_buffer_1"}
  ]
} {
  // CHECK-LABEL: module
  // CHECK-NOT: __udoncl_bound_
  // CHECK-NOT: __udoncl_target_
  // CHECK: udonasm.push "__udoncl_material_arg_1"
  // CHECK-NEXT: udonasm.push "__0_arg0__param"
  // CHECK-NEXT: udonasm.extern "UnityEngineMaterial.__SetFloat__SystemString_SystemSingle__SystemVoid"
  // CHECK: udonasm.push "__udoncl_material_arg_1"
  // CHECK-NEXT: udonasm.push "__0_arg0__param"
  // CHECK-NEXT: udonasm.extern "UnityEngineMaterial.__SetFloat__SystemString_SystemSingle__SystemVoid"
  func.func @mixed(%scale: f32) attributes {vrc.export_id = 1 : i32} {
    %output_id = arith.constant 1 : i32
    %bytes = arith.constant 4 : i32
    %output, %output_status = vrc.buffer_alloc %output_id, %bytes {
      format = "rgba8_unorm", usage = "storage"
    } : (i32, i32) -> (!vrc.buffer, i32)
    %desc0 = vrc.dispatch_desc_create %scale {
      kernel_id = 1 : i32, scalar_kinds = ["f32"]
    } : (f32) -> !vrc.dispatch_desc
    %event0 = vrc.dispatch %desc0(%output) {
      argument_kinds = ["buffer", "f32"],
      resource_accesses = ["write"],
      global_offset = array<i32: 0, 0, 0>,
      global_size = array<i32: 1, 1, 1>,
      kernel_id = 1 : i32,
      local_size = array<i32: 1, 1, 1>,
      vrc.unity_dispatch_material_slot = "__udoncl_kernel_material_1",
      vrc.unity_output_resource_index = 0 : i32,
      vrc.unity_resource_heap_slots = ["__udoncl_buffer_1"]
    } : (!vrc.dispatch_desc, !vrc.buffer) -> !vrc.event
    %desc1 = vrc.dispatch_desc_create %scale {
      kernel_id = 1 : i32, scalar_kinds = ["f32"]
    } : (f32) -> !vrc.dispatch_desc
    %event1 = vrc.dispatch %desc1(%output) {
      argument_kinds = ["buffer", "f32"],
      resource_accesses = ["write"],
      global_offset = array<i32: 0, 0, 0>,
      global_size = array<i32: 1, 1, 1>,
      kernel_id = 1 : i32,
      local_size = array<i32: 1, 1, 1>,
      vrc.unity_dispatch_material_slot = "__udoncl_kernel_material_1",
      vrc.unity_output_resource_index = 0 : i32,
      vrc.unity_resource_heap_slots = [""]
    } : (!vrc.dispatch_desc, !vrc.buffer) -> !vrc.event
    return
  }
}
