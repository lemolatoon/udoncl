// RUN: %vrc-opt --split-input-file --verify-diagnostics --vrc-prove-synthesized-gather-kernels --vrc-lower-gather-device %s

// A conditional output store needs vrc.output_granule, which must divide the
// output words, and with four words per texel the granule must be whole
// texels: a fragment writes one texel or nothing.

gpu.module @kernels {
  gpu.func @no_granule(%in: memref<16xf32>, %out: memref<16xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 16, 1, 1>, vrc.kernel_id = 3 : i32, vrc.profile = "gather-f32", vrc.synthesized_kernel, vrc.words = 16 : i32} {
    %x = gpu.global_id x
    %i = arith.index_cast %x : index to i32
    %c8 = arith.constant 8 : i32
    %keep = arith.cmpi ult, %i, %c8 : i32
    %value = memref.load %in[%x] : memref<16xf32>
    scf.if %keep {
      // expected-error @below {{the store must write the output buffer at the top level of the kernel, or end a top-level scf.if of a kernel with vrc.output_granule}}
      memref.store %value, %out[%x] : memref<16xf32>
    }
    gpu.return
  }
}

// -----

gpu.module @kernels {
  // expected-error @below {{vrc.output_granule must be a positive divisor of vrc.words}}
  gpu.func @uneven_granule(%in: memref<16xf32>, %out: memref<16xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 16, 1, 1>, vrc.kernel_id = 3 : i32, vrc.output_granule = 3 : i64, vrc.profile = "gather-f32", vrc.synthesized_kernel, vrc.words = 16 : i32} {
    %x = gpu.global_id x
    %value = memref.load %in[%x] : memref<16xf32>
    memref.store %value, %out[%x] : memref<16xf32>
    gpu.return
  }
}

// -----

gpu.module @kernels {
  gpu.func @store_with_else(%in: memref<16xf32>, %out: memref<16xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 16, 1, 1>, vrc.kernel_id = 3 : i32, vrc.output_granule = 8 : i64, vrc.profile = "gather-f32", vrc.synthesized_kernel, vrc.words = 16 : i32} {
    %x = gpu.global_id x
    %i = arith.index_cast %x : index to i32
    %c8 = arith.constant 8 : i32
    %keep = arith.cmpi ult, %i, %c8 : i32
    %value = memref.load %in[%x] : memref<16xf32>
    scf.if %keep {
      // expected-error @below {{the store must write the output buffer at the top level of the kernel}}
      memref.store %value, %out[%x] : memref<16xf32>
    } else {
    }
    gpu.return
  }
}

// -----

gpu.module @kernels {
  // expected-error @below {{partial-output kernel writes runs of 2 words, not whole texels of Device texel format rgba32_float}}
  gpu.func @partial_texel(%in: memref<16xf32>, %out: memref<16xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 16, 1, 1>, vrc.kernel_id = 3 : i32, vrc.output_granule = 2 : i64, vrc.profile = "gather-f32", vrc.synthesized_kernel, vrc.texel_format = "rgba32_float", vrc.words = 16 : i32} {
    %x = gpu.global_id x
    %i = arith.index_cast %x : index to i32
    %c2 = arith.constant 2 : i32
    %keep = arith.cmpi ult, %i, %c2 : i32
    %value = memref.load %in[%x] : memref<16xf32>
    scf.if %keep {
      memref.store %value, %out[%x] : memref<16xf32>
    }
    gpu.return
  }
}
