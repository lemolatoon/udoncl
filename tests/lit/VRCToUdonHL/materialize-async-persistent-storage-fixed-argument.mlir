// RUN: %vrc-opt --vrc-lower-host-to-async --async-func-to-async-runtime --vrc-prepare-udon-async-runtime --vrc-lower-udon-async-runtime --vrc-discover-async-persistent-storage --vrc-materialize-async-persistent-storage --vrc-discover-host-rematerialization --vrc-allocate-host-storage %s | %FileCheck %s

// An exported entry argument used after a suspension lives in fixed ABI
// parameter storage that only the export's own invocation writes. It needs no
// colorable persistent slot: the resumed code reads the argument directly and
// no persistent request is planned for it.

// CHECK: vrc.async_persistent_direct_requests = []
// CHECK-LABEL: func.func @run(%arg0: !vrc.event, %arg1: i32)
// CHECK-NOT: udonhl.spill_load
// CHECK: arith.addi %{{[0-9]+}}, %arg1 : i32
module {
  func.func private @wait_leaf(%event: !vrc.event) -> i32 attributes {
    vrc.async_effect = "may_suspend"
  } {
    %status = vrc.await %event : (!vrc.event) -> i32
    return %status : i32
  }

  func.func @run(%event: !vrc.event, %limit: i32) -> i32 attributes {
    vrc.export_id = 1 : i32,
    vrc.parameter_names = ["event", "limit"],
    vrc.public_name = "Run"
  } {
    %status = func.call @wait_leaf(%event) : (!vrc.event) -> i32
    %result = arith.addi %status, %limit : i32
    return %result : i32
  }
}
