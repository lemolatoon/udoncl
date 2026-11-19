// RUN: not %vrc-opt --vrc-lower-udonhl-to-udonasm %s 2>&1 | %FileCheck %s

// A coroutine entered by the dispatcher that enters the dispatcher again is a
// recursive activation of static return slots. Reject it at compile time
// instead of emitting code that loops once the outer return slot is reused.

// CHECK: error: recursive internal Udon call graph has no static slot plan
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
    %f = arith.constant 1 : i32
    %p = arith.constant 2 : i32
    call @__udoncl_async_resume_dispatch(%f, %p) : (i32, i32) -> ()
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
