// RUN: %vrc-opt --vrc-lower-udonhl-to-udonasm %s | %FileCheck %s

// The generated binding executes one published host-transfer command after
// the Udon event yields and then clears __udoncl_async_command_kind. A second
// upload or readback published before that would silently overwrite the
// unexecuted command (the TinyLlama conversation lost its input_ids and
// attention_mask uploads this way). Publishing therefore checks that the slot
// is idle and otherwise yields CL_INVALID_OPERATION (-59) as the event.

// CHECK-DAG: initial = "0", name = "__udoncl_async_command_idle_kind"
// CHECK-DAG: initial = "-59", name = "__udoncl_async_command_overwrite_status"
// CHECK-LABEL: udonasm.push "__udoncl_async_command_kind"
// CHECK-NEXT: udonasm.push "__udoncl_async_command_idle_kind"
// CHECK-NEXT: udonasm.push "__udoncl_async_command_slot_idle"
// CHECK-NEXT: udonasm.extern "SystemInt32.__op_Equality__SystemInt32_SystemInt32__SystemBoolean"
// CHECK-NEXT: udonasm.push "__udoncl_async_command_slot_idle"
// CHECK-NEXT: udonasm.jump_if_false [[OCCUPIED:[0-9]+]]
// CHECK: udonasm.push "__vrc_f7_upload_command"
// CHECK-NEXT: udonasm.push "__udoncl_async_command_kind"
// CHECK-NEXT: udonasm.copy
// CHECK-NEXT: udonasm.push "__udoncl_async_event"
// CHECK-NEXT: udonasm.push "[[EVENT:[^"]+]]"
// CHECK-NEXT: udonasm.copy
// CHECK-NEXT: udonasm.jump
// CHECK-NEXT: udonasm.push "__udoncl_async_command_overwrite_status"
// CHECK-NEXT: udonasm.push "[[EVENT]]"
// CHECK-NEXT: udonasm.copy
// CHECK: udonasm.extern "SystemInt32.__op_Equality__SystemInt32_SystemInt32__SystemBoolean"
// CHECK-NEXT: udonasm.push "__udoncl_async_command_slot_idle"
// CHECK-NEXT: udonasm.jump_if_false
// CHECK: udonasm.push "__vrc_f7_readback_command"
// CHECK-NEXT: udonasm.push "__udoncl_async_command_kind"
// CHECK: udonasm.push "__udoncl_async_command_overwrite_status"
module attributes {
  vrc.host_stage = "udonhl",
  vrc.unity_buffer_pool = [
    {id = 1 : i32, logical_bytes = 4 : i32, heap_slot = "__udoncl_buffer_1"}
  ]
} {
  func.func @transfer() -> i32 attributes {
    vrc.async_effect = "may_suspend",
    vrc.export_id = 7 : i32
  } {
    %id = arith.constant 1 : i32
    %bytes = arith.constant 4 : i32
    %zero = arith.constant 0 : i32
    %pointer = arith.constant 16 : i32
    %buffer, %status = vrc.buffer_alloc %id, %bytes {
      format = "rgba8_unorm", usage = "storage"
    } : (i32, i32) -> (!vrc.buffer, i32)
    %loaded_id = arith.addi %id, %zero : i32
    %host_pointer = arith.addi %pointer, %zero : i32
    %recovered = builtin.unrealized_conversion_cast %loaded_id : i32 to !vrc.buffer
    %uploaded = vrc.buffer_upload %recovered, %zero, %bytes, %host_pointer : (!vrc.buffer, i32, i32, i32) -> !vrc.event
    %upload_status = vrc.await %uploaded : (!vrc.event) -> i32
    %read = vrc.buffer_readback %buffer, %zero, %bytes, %pointer : (!vrc.buffer, i32, i32, i32) -> !vrc.event
    %read_status = vrc.await %read : (!vrc.event) -> i32
    %result = arith.addi %upload_status, %read_status : i32
    return %result : i32
  }
}
