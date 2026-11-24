// RUN: %vrc-opt --split-input-file --verify-diagnostics --vrc-describe-unity-device-kernels %s

// A kernel id without a library contract or a user kernel is not described.
func.func @unknown(%x: !vrc.buffer, %y: !vrc.buffer) {
  %desc = vrc.dispatch_desc_create {kernel_id = 77 : i32, scalar_kinds = []} : () -> !vrc.dispatch_desc
  // expected-error @below {{the Unity Host dispatches Device kernel id 77, which is neither a UdonCL Device library kernel nor an OpenCL C kernel of the Host's Device libraries}}
  %event = vrc.dispatch %desc(%x, %y) {
    argument_kinds = ["buffer", "buffer"], resource_accesses = ["read", "write"],
    global_offset = array<i32: 0, 0, 0>, global_size = array<i32: 1, 1, 1>,
    kernel_id = 77 : i32, local_size = array<i32: 1, 1, 1>
  } : (!vrc.dispatch_desc, !vrc.buffer, !vrc.buffer) -> !vrc.event
  return
}

// -----

// A constant scalar outside the library contract: the arena SGEMM walks at
// most 8192 K elements.
func.func @outside(%a: !vrc.buffer, %b: !vrc.buffer, %c: !vrc.buffer) {
  %one = arith.constant 1 : i32
  %k = arith.constant 8193 : i32
  %zero = arith.constant 0 : i32
  %alpha = arith.constant 1.0 : f32
  %desc = vrc.dispatch_desc_create %one, %one, %k, %zero, %zero, %alpha {
    kernel_id = 221 : i32, scalar_kinds = ["i32", "i32", "i32", "i32", "i32", "f32"]
  } : (i32, i32, i32, i32, i32, f32) -> !vrc.dispatch_desc
  // expected-error @below {{a dispatch of Device kernel udonblas_arena_sgemm (id 221) passes 8193 as argument 5, outside its library contract [1, 8192]}}
  %event = vrc.dispatch %desc(%a, %b, %c) {
    argument_kinds = ["buffer", "buffer", "buffer", "i32", "i32", "i32", "i32", "i32", "f32"],
    resource_accesses = ["read", "read", "write"],
    global_offset = array<i32: 0, 0, 0>, global_size = array<i32: 1, 1, 1>,
    kernel_id = 221 : i32, local_size = array<i32: 1, 1, 1>
  } : (!vrc.dispatch_desc, !vrc.buffer, !vrc.buffer, !vrc.buffer) -> !vrc.event
  return
}

// -----

// The dispatch sites of one kernel agree on its launch.
func.func @disagree(%x: !vrc.buffer, %y: !vrc.buffer) {
  %desc = vrc.dispatch_desc_create {kernel_id = 222 : i32, scalar_kinds = []} : () -> !vrc.dispatch_desc
  %first = vrc.dispatch %desc(%x, %y) {
    argument_kinds = ["buffer", "buffer"], resource_accesses = ["read", "write"],
    global_offset = array<i32: 0, 0, 0>, global_size = array<i32: 1, 1, 1>,
    kernel_id = 222 : i32, local_size = array<i32: 1, 1, 1>
  } : (!vrc.dispatch_desc, !vrc.buffer, !vrc.buffer) -> !vrc.event
  // expected-error @below {{the dispatches of Device kernel udononnx_unary_f32 (id 222) disagree on its argument kinds or NDRange}}
  %second = vrc.dispatch %desc(%x, %y) {
    argument_kinds = ["buffer", "buffer"], resource_accesses = ["read", "write"],
    global_offset = array<i32: 0, 0, 0>, global_size = array<i32: 2, 1, 1>,
    kernel_id = 222 : i32, local_size = array<i32: 1, 1, 1>
  } : (!vrc.dispatch_desc, !vrc.buffer, !vrc.buffer) -> !vrc.event
  return
}
