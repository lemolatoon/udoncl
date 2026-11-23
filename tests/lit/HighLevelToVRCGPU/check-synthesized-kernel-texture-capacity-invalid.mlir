// RUN: %vrc-opt --split-input-file --verify-diagnostics --vrc-check-synthesized-kernel-texture-capacity="buffer-widths=4,4 buffer-heights=1,1" %s
// RUN: not %vrc-opt --vrc-check-synthesized-kernel-texture-capacity="buffer-widths=4,4 buffer-heights=1" %s 2>&1 | %FileCheck %s --check-prefix=MISMATCH

// MISMATCH: error: buffer-widths and buffer-heights must list the same number of Unity buffers

// A synthesized kernel that writes more words than the 4x1 textures the Unity
// buffers share hold is rejected.

module {
  gpu.module @device_gpu_4096 {
    // expected-error @below {{synthesized kernel @scale writes 8 words but the Unity buffers are 4x1 textures}}
    gpu.func @scale(%in: memref<8xf32>, %out: memref<8xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 4, 2, 1>, vrc.kernel_id = 4096 : i32, vrc.synthesized_kernel, vrc.words = 8 : i32} {
      gpu.return
    }
  }
}
