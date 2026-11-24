// RUN: %vrc-opt --split-input-file --vrc-describe-unity-device-kernels="synthesized-kernel-ids=4096 user-kernel-ids=600,601 user-kernel-names=scale,fill" %s | %FileCheck %s

// A library kernel takes its launch from its dispatch sites and its access
// contract from its library contract: the strided-batched arena SGEMM bounds
// m, n, k, its strides, its batches, its operations and its first K term.
// A synthesized kernel has no entry. Entries are ordered by kernel id, and a
// description of an earlier run is replaced.

// CHECK-LABEL: module attributes
// CHECK-SAME: vrc.unity_device_kernels = [{argument_bounds = [{index = 3 : i64, max = 16384 : i64, min = 1 : i64}, {index = 4 : i64, max = 16384 : i64, min = 1 : i64}, {index = 5 : i64, max = 8192 : i64, min = 1 : i64}, {index = 6 : i64, max = 131072 : i64, min = 0 : i64}, {index = 7 : i64, max = 131072 : i64, min = 0 : i64}, {index = 8 : i64, max = 16384 : i64, min = 1 : i64}, {index = 9 : i64, max = 16384 : i64, min = 1 : i64}, {index = 10 : i64, max = 1 : i64, min = 0 : i64}, {index = 11 : i64, max = 1 : i64, min = 0 : i64}, {index = 13 : i64, max = 8192 : i64, min = 0 : i64}], argument_kinds = ["buffer", "buffer", "buffer", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "f32", "i32"], fragment_cost = {active = [3, 4, 9], loads = 0 : i64, per_argument = [{index = 5 : i64, loads = 2 : i64}]}, global_offset = [0, 0, 0], global_size = [8192, 2, 1], input_words = [131072, 131072], kernel = "udonblas_arena_sgemm_strided_batched", kernel_id = 237 : i64, local_size = [1, 1, 1], profile = "gather-f32", schema_version = 1 : i64, words = 16384 : i64}]
module attributes {vrc.unity_device_kernels = []} {
  func.func @product(%a: !vrc.buffer, %b: !vrc.buffer, %c: !vrc.buffer) {
    %m = arith.constant 4 : i32
    %k = arith.constant 64 : i32
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %alpha = arith.constant 1.0 : f32
    %desc = vrc.dispatch_desc_create %m, %m, %k, %zero, %zero, %m, %one, %zero, %zero, %alpha, %zero {
      kernel_id = 237 : i32,
      scalar_kinds = ["i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "f32", "i32"]
    } : (i32, i32, i32, i32, i32, i32, i32, i32, i32, f32, i32) -> !vrc.dispatch_desc
    %event = vrc.dispatch %desc(%a, %b, %c) {
      argument_kinds = ["buffer", "buffer", "buffer", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "f32", "i32"],
      resource_accesses = ["read", "read", "write"],
      global_offset = array<i32: 0, 0, 0>,
      global_size = array<i32: 8192, 2, 1>,
      kernel_id = 237 : i32,
      local_size = array<i32: 1, 1, 1>
    } : (!vrc.dispatch_desc, !vrc.buffer, !vrc.buffer, !vrc.buffer) -> !vrc.event
    %fused_desc = vrc.dispatch_desc_create {
      kernel_id = 4096 : i32, scalar_kinds = []
    } : () -> !vrc.dispatch_desc
    %fused = vrc.dispatch %fused_desc(%a, %c) {
      argument_kinds = ["buffer", "buffer"],
      resource_accesses = ["read", "write"],
      global_offset = array<i32: 0, 0, 0>,
      global_size = array<i32: 8, 1, 1>,
      kernel_id = 4096 : i32,
      local_size = array<i32: 1, 1, 1>
    } : (!vrc.dispatch_desc, !vrc.buffer, !vrc.buffer) -> !vrc.event
    return
  }
}

// -----

// A user OpenCL C kernel has only its launch; one that reads fewer than two
// buffers is one invocation, not a gather.

// CHECK-LABEL: module attributes
// CHECK-SAME: vrc.unity_device_kernels = [{argument_kinds = ["buffer", "buffer", "i32"], global_offset = [0, 0, 0], global_size = [4, 2, 1], kernel = "scale", kernel_id = 600 : i64, local_size = [1, 1, 1], profile = "gather-f32", schema_version = 1 : i64, words = 8 : i64}, {argument_kinds = ["buffer", "i32"], global_offset = [0, 0, 0], global_size = [1, 1, 1], kernel = "fill", kernel_id = 601 : i64, local_size = [1, 1, 1], schema_version = 1 : i64}]
module {
  func.func @user(%x: !vrc.buffer, %y: !vrc.buffer, %count: i32) {
    %scale_desc = vrc.dispatch_desc_create %count {
      kernel_id = 600 : i32, scalar_kinds = ["i32"]
    } : (i32) -> !vrc.dispatch_desc
    %scale = vrc.dispatch %scale_desc(%x, %y) {
      argument_kinds = ["buffer", "buffer", "i32"],
      resource_accesses = ["read", "write"],
      global_offset = array<i32: 0, 0, 0>,
      global_size = array<i32: 4, 2, 1>,
      kernel_id = 600 : i32,
      local_size = array<i32: 1, 1, 1>
    } : (!vrc.dispatch_desc, !vrc.buffer, !vrc.buffer) -> !vrc.event
    %fill_desc = vrc.dispatch_desc_create %count {
      kernel_id = 601 : i32, scalar_kinds = ["i32"]
    } : (i32) -> !vrc.dispatch_desc
    %fill = vrc.dispatch %fill_desc(%y) {
      argument_kinds = ["buffer", "i32"],
      resource_accesses = ["write"],
      global_offset = array<i32: 0, 0, 0>,
      global_size = array<i32: 1, 1, 1>,
      kernel_id = 601 : i32,
      local_size = array<i32: 1, 1, 1>
    } : (!vrc.dispatch_desc, !vrc.buffer) -> !vrc.event
    return
  }
}

// -----

// A module without dispatches has an empty description.

// CHECK-LABEL: module attributes {vrc.unity_device_kernels = []}
module {
}
