// RUN: %vrc-opt --vrc-assign-synthesized-kernel-ids="first-id=4 reserved-ids=4,6" %s | %FileCheck %s
// RUN: %vrc-opt --vrc-assign-synthesized-kernel-ids %s | %FileCheck %s --check-prefix=DEFAULT
// RUN: %vrc-opt --vrc-assign-synthesized-kernel-ids="first-id=4 reserved-ids=4,6" --vrc-lower-opencl-aot-intrinsics %s | %FileCheck %s --check-prefix=LOWERED

// Dispatched synthesized kernels get ids in module order, starting at
// first-id and skipping reserved ids and the id an existing kernel already
// holds (5). Each one is isolated as gpu.module @device_gpu_<id>, and the
// declarations that named it describe the launch themselves. The uncalled
// declaration, the kernel nothing dispatches and both carriers are gone; the
// library kernel is untouched.

// CHECK: gpu.module @library {
// CHECK-NEXT: gpu.func @library_kernel
// CHECK-SAME: vrc.kernel_id = 5 : i32

// CHECK: gpu.module @device_gpu_7 {
// CHECK-NEXT: gpu.func @scale(%{{.+}}: memref<4xf32>, %{{.+}}: memref<8xf32>, %{{.+}}: f32, %{{.+}}: i32) kernel
// CHECK-SAME: vrc.kernel_id = 7 : i32
// CHECK-SAME: vrc.synthesized_kernel

// CHECK: gpu.module @device_gpu_8 {
// CHECK-NEXT: gpu.func @copy(
// CHECK-SAME: vrc.kernel_id = 8 : i32

// CHECK-NOT: gpu.module
// CHECK-NOT: @unused_kernel
// CHECK-NOT: @never_called

// CHECK: func.func private @__vrc_opencl_dispatch_copy(i32, i32) -> i32 attributes {vrc.device_dispatch = {buffer_count = 2 : i32, global_size = array<i32: 4, 2, 1>, kernel_id = 8 : i32}}
// CHECK: func.func private @__vrc_opencl_dispatch_first(i32, i32, f32, i32) -> i32 attributes {vrc.device_dispatch = {buffer_count = 2 : i32, fragment_cost = {loads = 2 : i64}, global_size = array<i32: 8, 1, 1>, kernel_id = 7 : i32}}
// CHECK: func.func private @second_name_of_scale(i32, i32, f32, i32) -> i32 attributes {vrc.device_dispatch = {buffer_count = 2 : i32, fragment_cost = {loads = 2 : i64}, global_size = array<i32: 8, 1, 1>, kernel_id = 7 : i32}}
// CHECK-NOT: vrc.synthesized_dispatch

// DEFAULT: gpu.module @device_gpu_1 {
// DEFAULT-NEXT: gpu.func @scale(
// DEFAULT: gpu.module @device_gpu_2 {
// DEFAULT-NEXT: gpu.func @copy(

// LOWERED-LABEL: func.func @host(
// LOWERED: vrc.dispatch_desc_create %{{.+}}, %{{.+}} {kernel_id = 7 : i32, scalar_kinds = ["f32", "i32"]}
// LOWERED: vrc.dispatch %{{.+}}(%{{.+}}, %{{.+}}) {argument_kinds = ["buffer", "buffer", "f32", "i32"], global_offset = array<i32: 0, 0, 0>, global_size = array<i32: 8, 1, 1>, kernel_id = 7 : i32, local_size = array<i32: 1, 1, 1>, resource_accesses = ["read", "write"], vrc.fragment_cost = {loads = 2 : i64}}
// LOWERED: vrc.dispatch %{{.+}}(%{{.+}}, %{{.+}}) {{.*}}kernel_id = 7 : i32
// LOWERED: vrc.dispatch %{{.+}}(%{{.+}}, %{{.+}}) {argument_kinds = ["buffer", "buffer"], global_offset = array<i32: 0, 0, 0>, global_size = array<i32: 4, 2, 1>, kernel_id = 8 : i32, local_size = array<i32: 1, 1, 1>, resource_accesses = ["read", "write"]}
module {
  gpu.module @library {
    gpu.func @library_kernel(%in: memref<4xf32>, %out: memref<4xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 4, 1, 1>, vrc.kernel_id = 5 : i32, vrc.profile = "gather-f32", vrc.words = 4 : i32} {
      gpu.return
    }
  }
  gpu.module @first_carrier {
    gpu.func @unused_kernel(%in: memref<4xf32>, %out: memref<4xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 4, 1, 1>, vrc.profile = "gather-f32", vrc.synthesized_kernel, vrc.words = 4 : i32} {
      gpu.return
    }
    gpu.func @scale(%in: memref<4xf32>, %out: memref<8xf32>, %factor: f32, %count: i32) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 8, 1, 1>, vrc.fragment_cost = {loads = 2 : i64}, vrc.profile = "gather-f32", vrc.synthesized_kernel, vrc.words = 8 : i32} {
      gpu.return
    }
  }
  gpu.module @second_carrier {
    gpu.func @copy(%in: memref<8xi32>, %out: memref<8xi32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 4, 2, 1>, vrc.profile = "gather-f32", vrc.synthesized_kernel, vrc.words = 8 : i32} {
      gpu.return
    }
  }
  func.func private @__vrc_opencl_dispatch_copy(i32, i32) -> i32 attributes {vrc.synthesized_dispatch = @second_carrier::@copy}
  func.func private @__vrc_opencl_dispatch_first(i32, i32, f32, i32) -> i32 attributes {vrc.synthesized_dispatch = @first_carrier::@scale}
  func.func private @second_name_of_scale(i32, i32, f32, i32) -> i32 attributes {vrc.synthesized_dispatch = @first_carrier::@scale}
  func.func private @never_called(i32, i32) -> i32 attributes {vrc.synthesized_dispatch = @first_carrier::@unused_kernel}
  func.func @host(%a: i32, %b: i32, %factor: f32, %count: i32) -> i32 {
    %first = func.call @__vrc_opencl_dispatch_first(%a, %b, %factor, %count) : (i32, i32, f32, i32) -> i32
    %second = func.call @second_name_of_scale(%b, %a, %factor, %count) : (i32, i32, f32, i32) -> i32
    %third = func.call @__vrc_opencl_dispatch_copy(%a, %b) : (i32, i32) -> i32
    return %third : i32
  }
}
