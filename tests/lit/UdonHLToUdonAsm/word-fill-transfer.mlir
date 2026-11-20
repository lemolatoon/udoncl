// RUN: %vrc-opt --vrc-lower-udonhl-to-udonasm %s | %FileCheck %s

// Device words written from the Host go through the transfer material, whose
// shader rewrites only the words [First, First + Count) of the destination
// and discards every other fragment.
//
// vrc.buffer_fill sets the word range, the destination width and the pattern
// period, loads the pattern words from the raw heap (exactly the constant
// pattern length here, otherwise until the period ends), Blits once and
// restores the material's whole-surface copy mode. The event is complete at
// issue.
//
// A vrc.buffer_upload with a transfer material writes a mutable buffer token,
// 1 to 128 bytes, at a word-aligned offset, size and Host pointer the same
// way, with the upload itself as the pattern; every other upload still
// publishes the guarded binding command.

// CHECK-LABEL: udonasm.push "__udoncl_buffer_pool"
// CHECK: udonasm.extern "SystemObjectArray.__Get__SystemInt32__SystemObject"
// CHECK: udonasm.push "[[FILL_DESTINATION:[^"]+]]"
// CHECK-NEXT: udonasm.push "__udoncl_transfer_width"
// CHECK-NEXT: udonasm.extern "UnityEngineRenderTexture.__get_width__SystemInt32"
// CHECK: udonasm.push "__udoncl_transfer_first"
// CHECK-NEXT: udonasm.extern "SystemInt32.__op_RightShift__SystemInt32_SystemInt32__SystemInt32"
// CHECK: udonasm.push "__udoncl_buffer_copy_material"
// CHECK-NEXT: udonasm.push "__udoncl_transfer_property_Mode"
// CHECK-NEXT: udonasm.push "[[TWO:[^"]+]]"
// CHECK-NEXT: udonasm.extern "UnityEngineMaterial.__SetInteger__SystemString_SystemInt32__SystemVoid"
// CHECK: udonasm.push "__udoncl_transfer_property_Period"
// CHECK: udonasm.extern "SystemInt32Array.__Get__SystemInt32__SystemInt32"
// CHECK: udonasm.push "__udoncl_transfer_property_Word0"
// CHECK-NEXT: udonasm.push "__udoncl_transfer_word"
// CHECK-NEXT: udonasm.extern "UnityEngineMaterial.__SetInteger__SystemString_SystemInt32__SystemVoid"
// CHECK-NEXT: udonasm.push "__udoncl_null_texture"
// CHECK-NEXT: udonasm.push "[[FILL_DESTINATION]]"
// CHECK-NEXT: udonasm.push "__udoncl_buffer_copy_material"
// CHECK-NEXT: udonasm.extern "VRCSDKBaseVRCGraphics.__Blit__UnityEngineTexture_UnityEngineRenderTexture_UnityEngineMaterial__SystemVoid"
// CHECK-NEXT: udonasm.push "__udoncl_buffer_copy_material"
// CHECK-NEXT: udonasm.push "__udoncl_transfer_property_Mode"
// CHECK-NEXT: udonasm.push "[[ZERO:[^"]+]]"
// CHECK-NEXT: udonasm.extern "UnityEngineMaterial.__SetInteger__SystemString_SystemInt32__SystemVoid"
// CHECK-NEXT: udonasm.push "__udoncl_completed_event"

// The upload guard: positive token, positive size, at most 128 bytes, and a
// word-aligned offset, size and pointer.
// CHECK: udonasm.extern "SystemInt32.__op_GreaterThan__SystemInt32_SystemInt32__SystemBoolean"
// CHECK-NEXT: udonasm.push "__udoncl_transfer_condition"
// CHECK-NEXT: udonasm.jump_if_false [[COMMAND:[0-9]+]]
// CHECK: udonasm.extern "SystemInt32.__op_LessThanOrEqual__SystemInt32_SystemInt32__SystemBoolean"
// CHECK: udonasm.extern "SystemInt32.__op_LogicalOr__SystemInt32_SystemInt32__SystemInt32"
// CHECK: udonasm.extern "SystemInt32.__op_LogicalAnd__SystemInt32_SystemInt32__SystemInt32"
// CHECK: udonasm.extern "SystemInt32.__op_Equality__SystemInt32_SystemInt32__SystemBoolean"
// CHECK: udonasm.push "__udoncl_transfer_property_Word1"
// CHECK: udonasm.extern "SystemInt32.__op_LessThan__SystemInt32_SystemInt32__SystemBoolean"
// CHECK-NEXT: udonasm.push "__udoncl_transfer_condition"
// CHECK-NEXT: udonasm.jump_if_false [[LOADED:[0-9]+]]
// CHECK: udonasm.push "__udoncl_transfer_property_Word31"
// CHECK: udonasm.extern "VRCSDKBaseVRCGraphics.__Blit__UnityEngineTexture_UnityEngineRenderTexture_UnityEngineMaterial__SystemVoid"
// CHECK: udonasm.push "__udoncl_completed_event"
// CHECK-NEXT: udonasm.push "[[UPLOAD_EVENT:[^"]+]]"
// CHECK-NEXT: udonasm.copy
// CHECK-NEXT: udonasm.jump [[DONE:[0-9]+]]
// CHECK-NEXT: udonasm.push "__udoncl_async_command_kind"
// CHECK: udonasm.push "__vrc_f7_upload_command"
// CHECK-NEXT: udonasm.push "__udoncl_async_command_kind"
// CHECK-NEXT: udonasm.copy
module attributes {
  vrc.host_stage = "udonhl",
  vrc.unity_buffer_pool = [
    {id = 1 : i32, logical_bytes = 64 : i32, heap_slot = "__udoncl_buffer_1"}
  ]
} {
  "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, initial_value = 0 : i32, linkage = "internal", storage_type = i32, sym_name = "pattern"}> : () -> ()
  func.func @transfer(%token: i32, %offset: i32, %size: i32) -> i32 attributes {
    vrc.async_effect = "may_suspend",
    vrc.export_id = 7 : i32
  } {
    %four = arith.constant 4 : i32
    %bytes = arith.constant 64 : i32
    %address = udonhl.address_of @pattern : !udonhl.ptr<0>
    %pointer = "udonhl.ptrtoint"(%address) : (!udonhl.ptr<0>) -> i32
    %buffer = builtin.unrealized_conversion_cast %token : i32 to !vrc.buffer
    %fill = vrc.buffer_fill %buffer, %offset, %bytes, %pointer, %four {vrc.unity_copy_material_slot = "__udoncl_buffer_copy_material"} : (!vrc.buffer, i32, i32, i32, i32) -> !vrc.event
    %uploaded = vrc.buffer_upload %buffer, %offset, %size, %pointer {vrc.unity_copy_material_slot = "__udoncl_buffer_copy_material"} : (!vrc.buffer, i32, i32, i32) -> !vrc.event
    %upload_status = vrc.await %uploaded : (!vrc.event) -> i32
    return %upload_status : i32
  }
}
