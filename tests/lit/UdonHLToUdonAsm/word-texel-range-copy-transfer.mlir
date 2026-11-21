// RUN: %vrc-opt --vrc-lower-udonhl-to-udonasm %s | %FileCheck %s

// With word texels (four words per texel) a range copy writes whole texels.
// It first copies the destination into the transfer scratch texture with the
// word pass (1) while the material's snapshot input is unbound, binds the
// scratch as _VRC_TransferPrevious, Blits the range with pass 1, and unbinds
// the snapshot again, so no Blit ever reads its own target: the next
// snapshot Blits into the scratch texture.

// CHECK: udonasm.push "[[DESTINATION:[^"]+copy_destination[^"]+_texture]]"
// CHECK-NEXT: udonasm.extern "SystemObjectArray.__Get__SystemInt32__SystemObject"
// The snapshot: destination -> scratch, pass 1.
// CHECK-NEXT: udonasm.push "[[DESTINATION]]"
// CHECK-NEXT: udonasm.push "__udoncl_transfer_scratch"
// CHECK-NEXT: udonasm.push "__udoncl_buffer_copy_material"
// CHECK-NEXT: udonasm.push "[[ONE:[^"]+]]"
// CHECK-NEXT: udonasm.extern "VRCSDKBaseVRCGraphics.__Blit__UnityEngineTexture_UnityEngineRenderTexture_UnityEngineMaterial_SystemInt32__SystemVoid"
// CHECK-NEXT: udonasm.push "__udoncl_buffer_copy_material"
// CHECK-NEXT: udonasm.push "__udoncl_transfer_property_Previous"
// CHECK-NEXT: udonasm.push "__udoncl_transfer_scratch"
// CHECK-NEXT: udonasm.extern "UnityEngineMaterial.__SetTexture__SystemString_UnityEngineTexture__SystemVoid"
// CHECK: udonasm.push "__udoncl_transfer_property_Delta"
// CHECK-NEXT: udonasm.push "__udoncl_transfer_delta"
// CHECK-NEXT: udonasm.extern "UnityEngineMaterial.__SetInteger__SystemString_SystemInt32__SystemVoid"
// The range Blit, then the snapshot input is unbound.
// CHECK-NEXT: udonasm.push "[[SOURCE:[^"]+]]"
// CHECK-NEXT: udonasm.push "[[DESTINATION]]"
// CHECK-NEXT: udonasm.push "__udoncl_buffer_copy_material"
// CHECK-NEXT: udonasm.push "[[ONE]]"
// CHECK-NEXT: udonasm.extern "VRCSDKBaseVRCGraphics.__Blit__UnityEngineTexture_UnityEngineRenderTexture_UnityEngineMaterial_SystemInt32__SystemVoid"
// CHECK-NEXT: udonasm.push "__udoncl_buffer_copy_material"
// CHECK-NEXT: udonasm.push "__udoncl_transfer_property_Previous"
// CHECK-NEXT: udonasm.push "__udoncl_null_texture"
// CHECK-NEXT: udonasm.extern "UnityEngineMaterial.__SetTexture__SystemString_UnityEngineTexture__SystemVoid"
// CHECK-NEXT: udonasm.push "__udoncl_buffer_copy_material"
// CHECK-NEXT: udonasm.push "__udoncl_transfer_property_Mode"
module attributes {
  vrc.host_stage = "udonhl",
  vrc.texel_format = "rgba32_float",
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
