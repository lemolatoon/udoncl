// RUN: %vrc-opt --vrc-lower-udonhl-to-udonasm %s | %FileCheck %s

// async_enter sets the entered coroutine's static return slot to the entering
// site, so it is an activation edge just like a call. The drain reaches the
// coroutine only through the unbanked dispatcher; the two must still receive
// different return-slot banks, otherwise the coroutine's return overwrites the
// drain's pending return address.

// CHECK-DAG: udonasm.jump_indirect "__vrc_internal_bank_0_return"
// CHECK-DAG: udonasm.jump_indirect "__vrc_internal_bank_1_return"
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
