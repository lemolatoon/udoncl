// RUN: %vrc-opt --split-input-file --verify-diagnostics --vrc-prove-synthesized-gather-kernels --vrc-lower-gather-device %s

// Four words per texel need texture rows of whole texels; an unknown texel
// format is rejected.

gpu.module @kernels {
  // expected-error @below {{Device texel format rgba32_float requires a texture row of whole texels, not 6 words}}
  gpu.func @odd_row(%in: memref<12xf32>, %out: memref<12xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 6, 2, 1>, vrc.kernel_id = 3 : i32, vrc.profile = "gather-f32", vrc.synthesized_kernel, vrc.texel_format = "rgba32_float", vrc.words = 12 : i32} {
    %x = gpu.global_id x
    %y = gpu.global_id y
    %six = arith.constant 6 : index
    %row = arith.muli %y, %six : index
    %linear = arith.addi %x, %row : index
    %value = memref.load %in[%linear] : memref<12xf32>
    memref.store %value, %out[%linear] : memref<12xf32>
    gpu.return
  }
}

// -----

gpu.module @kernels {
  // expected-error @below {{unsupported Device texel format rgba16_float}}
  gpu.func @unknown_format(%in: memref<8xf32>, %out: memref<8xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 8, 1, 1>, vrc.kernel_id = 4 : i32, vrc.profile = "gather-f32", vrc.synthesized_kernel, vrc.texel_format = "rgba16_float", vrc.words = 8 : i32} {
    %x = gpu.global_id x
    %value = memref.load %in[%x] : memref<8xf32>
    memref.store %value, %out[%x] : memref<8xf32>
    gpu.return
  }
}
