// RUN: %vrc-opt --vrc-schedule-udon-device-work="device-cost-budget=150 host-cost-budget=1000 fragment-weight=1" %s | %FileCheck %s

// A command that another command follows before any yield point is charged
// where it is issued, with a saturating sum; only the last one is charged at
// the yield point.
// CHECK-LABEL: func.func @batched(
// CHECK: vrc.dispatch %
// CHECK: arith.minsi
// CHECK: arith.minsi
// CHECK: vrc.dispatch %
// CHECK-NOT: arith.minsi
// CHECK: cf.cond_br
// CHECK: vrc.device_work_budget_checkpoint

module {
  func.func @batched(%a: !vrc.buffer, %b: !vrc.buffer) -> i32 attributes {vrc.async_effect = "may_suspend"} {
    %zero = arith.constant 0 : i32
    %desc = vrc.dispatch_desc_create %zero {kernel_id = 1 : i32, scalar_kinds = ["i32"]} : (i32) -> !vrc.dispatch_desc
    %first = vrc.dispatch %desc(%a, %b) {argument_kinds = ["buffer", "buffer", "i32"], resource_accesses = ["read", "write"], global_offset = array<i32: 0, 0, 0>, global_size = array<i32: 1, 1, 1>, kernel_id = 1 : i32, local_size = array<i32: 1, 1, 1>} : (!vrc.dispatch_desc, !vrc.buffer, !vrc.buffer) -> !vrc.event
    %desc2 = vrc.dispatch_desc_create %zero {kernel_id = 1 : i32, scalar_kinds = ["i32"]} : (i32) -> !vrc.dispatch_desc
    %second = vrc.dispatch %desc2(%b, %a) {argument_kinds = ["buffer", "buffer", "i32"], resource_accesses = ["read", "write"], global_offset = array<i32: 0, 0, 0>, global_size = array<i32: 1, 1, 1>, kernel_id = 1 : i32, local_size = array<i32: 1, 1, 1>} : (!vrc.dispatch_desc, !vrc.buffer, !vrc.buffer) -> !vrc.event
    %status = vrc.await %second : (!vrc.event) -> i32
    return %status : i32
  }
}
