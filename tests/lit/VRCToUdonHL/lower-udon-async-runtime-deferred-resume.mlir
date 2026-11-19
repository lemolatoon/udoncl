// RUN: %vrc-opt --vrc-lower-host-to-async --async-func-to-async-runtime --vrc-prepare-udon-async-runtime --vrc-lower-udon-async-runtime %s | %FileCheck %s

// A parent that resumes after one may-suspend child and then calls the same
// child again. Udon returns through static per-function slots, so the
// completing child must not enter its waiter (that would nest
// dispatcher -> coroutine -> dispatcher and overwrite an outer return slot
// once the resumed parent re-enters the child). It records the single ready
// waiter instead, and only the completion event drains it.

// CHECK-LABEL: func.func private @step(
// CHECK-NOT: call @__udoncl_async_resume
// CHECK: udonhl.address_of @__udoncl_async_ready_frame
// CHECK-NEXT: "udonhl.store"({{.*}}) {{.*}}vrc.async_ready_frame_store
// CHECK: udonhl.address_of @__udoncl_async_ready_pc
// CHECK-NEXT: "udonhl.store"({{.*}}) {{.*}}vrc.async_ready_pc_store
// CHECK-NOT: call @__udoncl_async_resume
// CHECK: return

// CHECK-LABEL: func.func @generate(
// CHECK-NOT: call @__udoncl_async_resume
// CHECK: call @step(
// CHECK-NOT: call @__udoncl_async_resume
// CHECK: call @step(
// CHECK-NOT: call @__udoncl_async_resume
// CHECK: return

// Every completion path enters at most one continuation and then drains.
// CHECK-LABEL: func.func @__udoncl_async_complete(
// CHECK: "udonhl.async_enter"()
// CHECK-NEXT: call @__udoncl_async_drain_ready() : () -> ()
// CHECK-NEXT: return
// CHECK-NOT: call @__udoncl_async_resume_object_

// CHECK-LABEL: func.func private @__udoncl_async_resume_dispatch(
// CHECK-NOT: call @

// CHECK-DAG: sym_name = "__udoncl_async_ready_frame"
// CHECK-DAG: sym_name = "__udoncl_async_ready_pc"

// The drain clears the one-element slot before entering the continuation and
// loops until the resumed code leaves nothing ready.
// CHECK-LABEL: func.func private @__udoncl_async_drain_ready() attributes {vrc.compiler_async_drain}
// CHECK: cf.br ^[[CHECK:bb[0-9]+]]
// CHECK: ^[[CHECK]]:
// CHECK: %[[FRAME:.*]] = "udonhl.load"
// CHECK: arith.cmpi ne, %[[FRAME]]
// CHECK: cf.cond_br
// CHECK: "udonhl.store"
// CHECK: "udonhl.store"
// CHECK: call @__udoncl_async_resume_dispatch(
// CHECK-NEXT: cf.br ^[[CHECK]]
// CHECK: return
module {
  func.func private @step(%event: !vrc.event, %value: i32) -> i32 attributes {
    vrc.async_effect = "may_suspend"
  } {
    %status = vrc.await %event : (!vrc.event) -> i32
    %result = arith.addi %value, %status : i32
    return %result : i32
  }

  func.func @generate(%event: !vrc.event, %value: i32) -> i32 attributes {
    vrc.export_id = 1 : i32,
    vrc.parameter_names = ["event", "value"],
    vrc.public_name = "Generate"
  } {
    %first = func.call @step(%event, %value)
      : (!vrc.event, i32) -> i32
    %second = func.call @step(%event, %first)
      : (!vrc.event, i32) -> i32
    return %second : i32
  }
}
