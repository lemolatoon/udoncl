// RUN: %vrc-opt --vrc-lower-opencl-aot-intrinsics %s | %FileCheck %s

// The OpenCL AOT runtime fills a word-aligned Device range through
// __vrc_opencl_buffer_fill(token, offset, bytes, pattern, pattern_bytes).
// It becomes a typed vrc.buffer_fill whose Host pattern pointer is an i32
// raw-heap address and whose event is the intrinsic's status token.

// CHECK-LABEL: func.func @reset(
// CHECK-SAME: %[[TOKEN:.+]]: i32, %[[PATTERN:.+]]: !udonhl.ptr<0>
// CHECK: %[[BUFFER:.+]] = builtin.unrealized_conversion_cast %[[TOKEN]] : i32 to !vrc.buffer
// CHECK: %[[ADDRESS:.+]] = "udonhl.ptrtoint"(%[[PATTERN]])
// CHECK: %[[EVENT:.+]] = vrc.buffer_fill %[[BUFFER]], %{{.+}}, %{{.+}}, %[[ADDRESS]], %{{.+}} : (!vrc.buffer, i32, i32, i32, i32) -> !vrc.event
// CHECK: builtin.unrealized_conversion_cast %[[EVENT]] : !vrc.event to i32
// CHECK-NOT: __vrc_opencl_buffer_fill
module {
  func.func private @__vrc_opencl_buffer_fill(i32, i32, i32, !udonhl.ptr<0>, i32) -> i32
  func.func @reset(%token: i32, %pattern: !udonhl.ptr<0>) -> i32 {
    %zero = arith.constant 0 : i32
    %bytes = arith.constant 64 : i32
    %four = arith.constant 4 : i32
    %event = func.call @__vrc_opencl_buffer_fill(%token, %zero, %bytes, %pattern, %four) : (i32, i32, i32, !udonhl.ptr<0>, i32) -> i32
    return %event : i32
  }
}
