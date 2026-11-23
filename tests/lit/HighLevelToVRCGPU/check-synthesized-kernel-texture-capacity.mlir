// RUN: %vrc-opt --vrc-check-synthesized-kernel-texture-capacity="buffer-widths=4,4 buffer-heights=2,2" %s | %FileCheck %s
// RUN: %vrc-opt --vrc-check-synthesized-kernel-texture-capacity="buffer-widths=4,2 buffer-heights=1,1" %s | %FileCheck %s
// RUN: %vrc-opt --vrc-check-synthesized-kernel-texture-capacity %s | %FileCheck %s

// Two 4x2 textures hold the 8 words of @scale exactly. Buffers of different
// shapes share no texture shape and bound nothing, nor does a library
// without buffers. The library kernel is never checked, and the module is
// left unchanged.

// CHECK: gpu.module @device_gpu_4096 {
// CHECK-NEXT: gpu.func @scale(
// CHECK-SAME: vrc.synthesized_kernel
// CHECK-SAME: vrc.words = 8 : i32
// CHECK: gpu.module @device_gpu_1 {
// CHECK-NEXT: gpu.func @library_kernel(
// CHECK-SAME: vrc.words = 64 : i32

module {
  gpu.module @device_gpu_4096 {
    gpu.func @scale(%in: memref<8xf32>, %out: memref<8xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 4, 2, 1>, vrc.kernel_id = 4096 : i32, vrc.synthesized_kernel, vrc.words = 8 : i32} {
      gpu.return
    }
  }
  gpu.module @device_gpu_1 {
    gpu.func @library_kernel(%in: memref<64xf32>, %out: memref<64xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 8, 8, 1>, vrc.kernel_id = 1 : i32, vrc.words = 64 : i32} {
      gpu.return
    }
  }
}
