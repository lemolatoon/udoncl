// RUN: %vrc-opt --vrc-lower-opencl-aot-intrinsics %s | %FileCheck %s

// The OpenCL AOT runtime calls __vrc_opencl_flush() after it publishes a
// non-blocking read. Returning to the binding submits the published transfer
// at its position in the in-order queue, and the Host does not wait for it:
// the call becomes a cooperative checkpoint, an await of an already complete
// event, and its function may suspend.

// CHECK-LABEL: func.func @read(
// CHECK-SAME: vrc.async_effect = "may_suspend"
// CHECK: %[[READ:.+]] = vrc.buffer_readback
// CHECK: %[[COMPLETE:.+]] = arith.constant 0 : i32
// CHECK: %[[EVENT:.+]] = builtin.unrealized_conversion_cast %[[COMPLETE]] : i32 to !vrc.event
// CHECK: %[[STATUS:.+]] = vrc.await %[[EVENT]] {vrc.cooperative_checkpoint} : (!vrc.event) -> i32
// CHECK: return %[[STATUS]] : i32
// CHECK-NOT: __vrc_opencl_flush
module {
  func.func private @__vrc_opencl_buffer_readback(i32, i32, i32, !udonhl.ptr<0>) -> i32
  func.func private @__vrc_opencl_flush() -> i32
  func.func @read(%token: i32, %destination: !udonhl.ptr<0>) -> i32 {
    %zero = arith.constant 0 : i32
    %bytes = arith.constant 4 : i32
    %pending = func.call @__vrc_opencl_buffer_readback(%token, %zero, %bytes, %destination) : (i32, i32, i32, !udonhl.ptr<0>) -> i32
    %status = func.call @__vrc_opencl_flush() : () -> i32
    return %status : i32
  }
}
