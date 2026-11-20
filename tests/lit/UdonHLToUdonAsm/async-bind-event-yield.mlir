// RUN: %vrc-opt --vrc-lower-udonhl-to-udonasm %s | %FileCheck %s

// A suspension publishes the async object it waits on in
// __udoncl_async_pending_object. A cooperative checkpoint depends on no
// transfer result, so it also sets __udoncl_async_pending_yield: the generated
// binding then resumes the Host while a published readback is still in
// flight. Any other suspension leaves the flag clear and waits for the
// outstanding transfer.

// CHECK-DAG: initial = "0", name = "__udoncl_async_pending_yield", public
// CHECK-LABEL: udonasm.push "__udoncl_async_pending_object"
// CHECK-NEXT: udonasm.copy
// CHECK-NEXT: udonasm.push "[[ONE:[^"]+]]"
// CHECK-NEXT: udonasm.push "__udoncl_async_pending_yield"
// CHECK-NEXT: udonasm.copy
// CHECK: udonasm.push "__udoncl_async_pending_object"
// CHECK-NEXT: udonasm.copy
// CHECK-NOT: udonasm.push "__udoncl_async_pending_yield"
module attributes {vrc.host_stage = "udon-static-v1"} {
  func.func @start() attributes {
    vrc.export_id = 8 : i32,
    vrc.parameter_names = [],
    vrc.public_name = "Start"
  } {
    call @continuation() : () -> ()
    return
  }
  func.func private @continuation() attributes {
    vrc.async_frame_id = 1 : i32
  } {
    %complete = arith.constant 0 : i32
    %checkpoint = builtin.unrealized_conversion_cast %complete : i32 to !vrc.event
    "udonhl.async_bind_event"(%checkpoint) <{object_id = 3 : i32}> {vrc.cooperative_checkpoint} : (!vrc.event) -> ()
    %token = arith.constant 1 : i32
    %transfer = builtin.unrealized_conversion_cast %token : i32 to !vrc.event
    "udonhl.async_bind_event"(%transfer) <{object_id = 4 : i32}> : (!vrc.event) -> ()
    cf.br ^resume
  ^resume:
    "udonhl.async_resume_point"() <{frame_id = 1 : i32, resume_pc = 2 : i32}> : () -> ()
    return
  }
  func.func private @__udoncl_async_resume_dispatch(%frame: i32, %pc: i32) attributes {
    vrc.compiler_async_dispatch
  } {
    "udonhl.async_enter"() <{frame_id = 1 : i32, resume_pc = 2 : i32}> : () -> ()
    return
  }
  func.func private @__udoncl_async_drain_ready() attributes {
    vrc.compiler_async_drain
  } {
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    call @__udoncl_async_resume_dispatch(%c1, %c2) : (i32, i32) -> ()
    return
  }
  func.func @__udoncl_async_complete(%object: i32, %status: i32) attributes {
    vrc.compiler_async_completion,
    vrc.udon_event = "__udoncl_async_complete"
  } {
    call @__udoncl_async_drain_ready() : () -> ()
    return
  }
}
