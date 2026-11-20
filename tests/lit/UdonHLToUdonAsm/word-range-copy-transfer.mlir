// RUN: %vrc-opt --vrc-lower-udonhl-to-udonasm %s | %FileCheck %s

// A vrc.buffer_copy marked vrc.unity_copy_word_range copies a word range
// between two Device buffers with one Blit through the transfer material's
// range-copy mode: the shader writes only the destination words
// [First, First + Count), reads source word (word + Delta) and discards every
// other fragment, so the destination is never read back. The buffers may be
// bound at run time; the source and destination offsets may differ. The
// material's whole-surface mode is restored after the Blit and the event is
// complete at issue.

// CHECK-LABEL: udonasm.push "__udoncl_readable_texture"
// CHECK: udonasm.push "[[DESTINATION:[^"]+copy_destination[^"]+_texture]]"
// CHECK-NEXT: udonasm.extern "SystemObjectArray.__Get__SystemInt32__SystemObject"
// CHECK-NEXT: udonasm.push "[[DESTINATION]]"
// CHECK-NEXT: udonasm.push "__udoncl_transfer_width"
// CHECK-NEXT: udonasm.extern "UnityEngineRenderTexture.__get_width__SystemInt32"
// CHECK: udonasm.push "__udoncl_transfer_first"
// CHECK-NEXT: udonasm.extern "SystemInt32.__op_RightShift__SystemInt32_SystemInt32__SystemInt32"
// CHECK: udonasm.push "__udoncl_transfer_count"
// CHECK-NEXT: udonasm.extern "SystemInt32.__op_RightShift__SystemInt32_SystemInt32__SystemInt32"
// CHECK: udonasm.push "__udoncl_transfer_delta"
// CHECK-NEXT: udonasm.extern "SystemInt32.__op_RightShift__SystemInt32_SystemInt32__SystemInt32"
// CHECK-NEXT: udonasm.push "__udoncl_transfer_delta"
// CHECK-NEXT: udonasm.push "__udoncl_transfer_first"
// CHECK-NEXT: udonasm.push "__udoncl_transfer_delta"
// CHECK-NEXT: udonasm.extern "SystemInt32.__op_Subtraction__SystemInt32_SystemInt32__SystemInt32"
// CHECK-NEXT: udonasm.push "__udoncl_buffer_copy_material"
// CHECK-NEXT: udonasm.push "__udoncl_transfer_property_Mode"
// CHECK-NEXT: udonasm.push "[[ONE:[^"]+]]"
// CHECK-NEXT: udonasm.extern "UnityEngineMaterial.__SetInteger__SystemString_SystemInt32__SystemVoid"
// CHECK: udonasm.push "__udoncl_transfer_property_Width"
// CHECK: udonasm.push "__udoncl_transfer_property_First"
// CHECK: udonasm.push "__udoncl_transfer_property_Count"
// CHECK: udonasm.push "__udoncl_transfer_property_Delta"
// CHECK-NEXT: udonasm.push "__udoncl_transfer_delta"
// CHECK-NEXT: udonasm.extern "UnityEngineMaterial.__SetInteger__SystemString_SystemInt32__SystemVoid"
// CHECK-NEXT: udonasm.push "[[SOURCE:[^"]+]]"
// CHECK-NEXT: udonasm.push "[[DESTINATION]]"
// CHECK-NEXT: udonasm.push "__udoncl_buffer_copy_material"
// CHECK-NEXT: udonasm.extern "VRCSDKBaseVRCGraphics.__Blit__UnityEngineTexture_UnityEngineRenderTexture_UnityEngineMaterial__SystemVoid"
// CHECK-NEXT: udonasm.push "__udoncl_buffer_copy_material"
// CHECK-NEXT: udonasm.push "__udoncl_transfer_property_Mode"
// CHECK-NEXT: udonasm.push "[[ZERO:[^"]+]]"
// CHECK-NEXT: udonasm.extern "UnityEngineMaterial.__SetInteger__SystemString_SystemInt32__SystemVoid"
// CHECK-NEXT: udonasm.push "__udoncl_completed_event"
module attributes {
  vrc.host_stage = "udonhl",
  vrc.unity_buffer_pool = [
    {id = 1 : i32, logical_bytes = 4 : i32, heap_slot = "__udoncl_buffer_1"},
    {id = 2 : i32, logical_bytes = 64 : i32, heap_slot = "__udoncl_buffer_2"}
  ]
} {
  func.func @transfer(%source_token: i32, %destination_token: i32, %source_offset: i32, %destination_offset: i32) -> i32 attributes {
    vrc.export_id = 7 : i32
  } {
    %bytes = arith.constant 4 : i32
    %source = builtin.unrealized_conversion_cast %source_token : i32 to !vrc.buffer
    %destination = builtin.unrealized_conversion_cast %destination_token : i32 to !vrc.buffer
    %copied = vrc.buffer_copy %source, %destination, %source_offset, %destination_offset, %bytes {vrc.unity_copy_material_slot = "__udoncl_buffer_copy_material", vrc.unity_copy_word_range} : (!vrc.buffer, !vrc.buffer, i32, i32, i32) -> !vrc.event
    %status = builtin.unrealized_conversion_cast %copied : !vrc.event to i32
    return %status : i32
  }
}
