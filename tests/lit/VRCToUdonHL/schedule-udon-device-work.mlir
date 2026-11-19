// RUN: %vrc-opt --vrc-schedule-udon-device-work="device-cost-budget=150 host-cost-budget=1000 fragment-weight=1" %s | %FileCheck %s
// RUN: %vrc-opt --vrc-schedule-udon-device-work="device-cost-budget=150 host-cost-budget=1000 fragment-weight=1" --vrc-lower-host-to-async --async-func-to-async-runtime --vrc-prepare-udon-async-runtime %s | %FileCheck %s --check-prefix=ASYNC
// RUN: not %vrc-opt --vrc-schedule-udon-device-work="device-cost-budget=0" %s 2>&1 | %FileCheck %s --check-prefix=BUDGET

// Device commands are complete at issue on the Udon queue: their awaits fold
// to CL_SUCCESS, so issuing them never suspends by itself. A Host transfer
// keeps its await and starts a new slice. Every command is charged its static
// Device and Host cost after it is issued, and each folded await becomes a
// cooperative checkpoint taken only when either slice budget is full.
//
// Device cost (fragment weight 1):
//  - first: 8x8 launch, no declaration, one read resource: 64 + 64 = 128;
//  - second: 4 launched fragments, declared over min(4, arg2 = 3) active
//    fragments with 1 + 4 * arg2 + 2 * arg3 loads; arg3 is not a constant and
//    uses its declared maximum 10: 4 + 3 * (1 + 12 + 20) = 103;
//  - copy: 4 bytes are one word, one fragment and one load: 2.
// Host cost, in estimated Udon instructions: 32 per kernel argument plus 32
// for the Blit, and 8 per Host operation of the block since the previous
// command (the command included):
//  - first: 3 arguments, 7 operations: 128 + 56 = 184;
//  - second: 4 arguments, 3 operations (await, descriptor, dispatch): 184;
//  - copy: 2 bound buffers, 2 operations (await, copy): 96 + 16 = 112.

// CHECK: module attributes {vrc.device_work_folded_awaits = 3 : i32, vrc.device_work_static_cost = {commands = 3 : i64, device = 233 : i64, host = 480 : i64}}
// CHECK-LABEL: func.func @run(
// CHECK: %[[UPLOAD:.+]] = vrc.buffer_upload
// CHECK: vrc.await %[[UPLOAD]]
// CHECK-NEXT: %[[RESET:.+]] = arith.constant 0 : i32
// CHECK-NEXT: %[[SLOT:.+]] = udonhl.address_of @__udoncl_slice_device_cost
// CHECK-NEXT: "udonhl.store"(%[[RESET]], %[[SLOT]])
// CHECK: %[[FIRST:.+]] = vrc.dispatch %
// CHECK-NEXT: arith.constant 128 : i32
// CHECK-NEXT: udonhl.address_of @__udoncl_slice_device_cost
// CHECK: arith.constant 184 : i32
// CHECK-NEXT: udonhl.address_of @__udoncl_slice_host_cost
// CHECK-NOT: vrc.await %[[FIRST]]
// CHECK: arith.cmpi sge
// CHECK: cf.cond_br
// CHECK: vrc.await %{{.+}} {vrc.cooperative_checkpoint, vrc.device_work_budget_checkpoint}
// CHECK: %[[SECOND:.+]] = vrc.dispatch %{{.+}}vrc.fragment_cost
// CHECK-NEXT: arith.constant 103 : i32
// CHECK: arith.constant 184 : i32
// CHECK-NEXT: udonhl.address_of @__udoncl_slice_host_cost
// CHECK-NOT: vrc.await %[[SECOND]]
// CHECK: vrc.device_work_budget_checkpoint
// CHECK: %[[COPY:.+]] = vrc.buffer_copy
// CHECK-NEXT: arith.constant 2 : i32
// CHECK: arith.constant 112 : i32
// CHECK-NEXT: udonhl.address_of @__udoncl_slice_host_cost
// CHECK-NOT: vrc.await %[[COPY]]
// CHECK: vrc.device_work_budget_checkpoint
// CHECK: return
// A function that cannot suspend keeps its ABI: no checkpoint and no charge.
// CHECK-LABEL: func.func @sync_issue(
// CHECK-NOT: __udoncl_slice_
// CHECK: vrc.dispatch %
// CHECK-NEXT: return
// CHECK: sym_name = "__udoncl_slice_device_cost"
// CHECK: sym_name = "__udoncl_slice_host_cost"

// ASYNC-COUNT-3: vrc.event_as_async{{.*}}vrc.cooperative_checkpoint
// ASYNC-NOT: vrc.event_as_async{{.*}}vrc.cooperative_checkpoint

// BUDGET: Device work slice budgets must be in [1, 2^30]
module {
  func.func @run(%a: !vrc.buffer, %b: !vrc.buffer, %pointer: i32) -> i32 attributes {
    vrc.async_effect = "may_suspend",
    vrc.export_id = 1 : i32,
    vrc.parameter_names = ["a", "b", "pointer"],
    vrc.public_name = "Run"
  } {
    %zero = arith.constant 0 : i32
    %three = arith.constant 3 : i32
    %four = arith.constant 4 : i32
    %upload = vrc.buffer_upload %a, %zero, %four, %pointer : (!vrc.buffer, i32, i32, i32) -> !vrc.event
    %uploaded = vrc.await %upload : (!vrc.event) -> i32
    %desc = vrc.dispatch_desc_create %zero {kernel_id = 1 : i32, scalar_kinds = ["i32"]} : (i32) -> !vrc.dispatch_desc
    %first = vrc.dispatch %desc(%a, %b) {argument_kinds = ["buffer", "buffer", "i32"], resource_accesses = ["read", "write"], global_offset = array<i32: 0, 0, 0>, global_size = array<i32: 8, 8, 1>, kernel_id = 1 : i32, local_size = array<i32: 1, 1, 1>} : (!vrc.dispatch_desc, !vrc.buffer, !vrc.buffer) -> !vrc.event
    %first_status = vrc.await %first : (!vrc.event) -> i32
    %desc2 = vrc.dispatch_desc_create %three, %pointer {kernel_id = 2 : i32, scalar_kinds = ["i32", "i32"]} : (i32, i32) -> !vrc.dispatch_desc
    %second = vrc.dispatch %desc2(%b, %a) {argument_kinds = ["buffer", "buffer", "i32", "i32"], resource_accesses = ["read", "write"], global_offset = array<i32: 0, 0, 0>, global_size = array<i32: 4, 1, 1>, kernel_id = 2 : i32, local_size = array<i32: 1, 1, 1>, vrc.fragment_cost = {loads = 1 : i64, active = [{index = 2 : i64, max = 50 : i64}], per_argument = [{index = 2 : i64, loads = 4 : i64, max = 50 : i64}, {index = 3 : i64, loads = 2 : i64, max = 10 : i64}]}} : (!vrc.dispatch_desc, !vrc.buffer, !vrc.buffer) -> !vrc.event
    %second_status = vrc.await %second : (!vrc.event) -> i32
    %copy = vrc.buffer_copy %a, %b, %zero, %zero, %four : (!vrc.buffer, !vrc.buffer, i32, i32, i32) -> !vrc.event
    %copy_status = vrc.await %copy : (!vrc.event) -> i32
    %s0 = arith.addi %uploaded, %first_status : i32
    %s1 = arith.addi %s0, %second_status : i32
    %s2 = arith.addi %s1, %copy_status : i32
    return %s2 : i32
  }

  func.func @sync_issue(%a: !vrc.buffer, %b: !vrc.buffer) {
    %zero = arith.constant 0 : i32
    %desc = vrc.dispatch_desc_create %zero {kernel_id = 1 : i32, scalar_kinds = ["i32"]} : (i32) -> !vrc.dispatch_desc
    %event = vrc.dispatch %desc(%a, %b) {argument_kinds = ["buffer", "buffer", "i32"], resource_accesses = ["read", "write"], global_offset = array<i32: 0, 0, 0>, global_size = array<i32: 1, 1, 1>, kernel_id = 1 : i32, local_size = array<i32: 1, 1, 1>} : (!vrc.dispatch_desc, !vrc.buffer, !vrc.buffer) -> !vrc.event
    return
  }
}
