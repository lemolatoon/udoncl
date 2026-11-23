// RUN: %vrc-opt --vrc-attach-device-resource-operand-contracts %s | %FileCheck %s

// A library entry point declares its fixed buffer count; a user OpenCL C
// kernel entry point names it. The last buffer is written.
// CHECK-LABEL: func.func private @__vrc_enqueue_udononnx_transpose_device_args
// CHECK-SAME: vrc.device_resource_operands = [{access = "read", capture = "none", index = 1 : i32}, {access = "write", capture = "none", index = 2 : i32}]
func.func private @__vrc_enqueue_udononnx_transpose_device_args(!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>, i32) -> i32
// CHECK-LABEL: func.func private @__vrc_enqueue_user_1024_3_saxpy
// CHECK-SAME: {access = "read", capture = "none", index = 1 : i32}, {access = "read", capture = "none", index = 2 : i32}, {access = "write", capture = "none", index = 3 : i32}
func.func private @__vrc_enqueue_user_1024_3_saxpy(!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>, i32, f32) -> i32
// Too few arguments, an unknown name or a definition get no contract.
// CHECK-LABEL: func.func private @__vrc_enqueue_user_1025_3_short
// CHECK-NOT: vrc.device_resource_operands
func.func private @__vrc_enqueue_user_1025_3_short(!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
// CHECK-LABEL: func.func private @unrelated
// CHECK-NOT: vrc.device_resource_operands
func.func private @unrelated(!udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
