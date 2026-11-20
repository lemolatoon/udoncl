// RUN: %vrc-opt --vrc-schedule-udon-device-work="device-cost-budget=1000 host-cost-budget=1000 fragment-weight=1" %s | %FileCheck %s

// A vrc.buffer_fill is one Blit of its destination words with no texel load,
// complete at issue like a dispatch: its await folds into a budget
// checkpoint, and it is charged
//  - Device cost: 64 bytes are 16 words, one fragment each at weight 1: 16;
//  - Host cost: 8 bound values (destination, range, width, period and pattern
//    words) plus the Blit at 32 each, and 4 operations (the fill and the
//    three constants before it) at 8 each: 288 + 32 = 320.
// An upload stays a Host transfer and keeps its suspending await.

// CHECK: module attributes {vrc.device_work_folded_awaits = 1 : i32, vrc.device_work_static_cost = {commands = 1 : i64, device = 16 : i64, host = 320 : i64}}
// CHECK-LABEL: func.func @reset(
// CHECK: %[[FILL:.+]] = vrc.buffer_fill
// CHECK-NEXT: arith.constant 16 : i32
// CHECK-NEXT: udonhl.address_of @__udoncl_slice_device_cost
// CHECK: arith.constant 320 : i32
// CHECK-NEXT: udonhl.address_of @__udoncl_slice_host_cost
// CHECK-NOT: vrc.await %[[FILL]]
// CHECK: vrc.await %{{.+}} {vrc.cooperative_checkpoint, vrc.device_work_budget_checkpoint}
// CHECK: %[[UPLOAD:.+]] = vrc.buffer_upload
// CHECK: vrc.await %[[UPLOAD]] :
module {
  func.func @reset(%buffer: !vrc.buffer, %pointer: i32) -> i32 attributes {
    vrc.async_effect = "may_suspend",
    vrc.export_id = 1 : i32,
    vrc.parameter_names = ["buffer", "pointer"],
    vrc.public_name = "Reset"
  } {
    %zero = arith.constant 0 : i32
    %four = arith.constant 4 : i32
    %bytes = arith.constant 64 : i32
    %fill = vrc.buffer_fill %buffer, %zero, %bytes, %pointer, %four : (!vrc.buffer, i32, i32, i32, i32) -> !vrc.event
    %filled = vrc.await %fill : (!vrc.event) -> i32
    %upload = vrc.buffer_upload %buffer, %zero, %four, %pointer : (!vrc.buffer, i32, i32, i32) -> !vrc.event
    %uploaded = vrc.await %upload : (!vrc.event) -> i32
    %status = arith.addi %filled, %uploaded : i32
    return %status : i32
  }
}
