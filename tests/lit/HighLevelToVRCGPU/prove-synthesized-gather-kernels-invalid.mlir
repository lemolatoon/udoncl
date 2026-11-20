// RUN: %vrc-opt --split-input-file --verify-diagnostics --vrc-prove-synthesized-gather-kernels %s

// A synthesized gather kernel is rejected, with a diagnostic on the offending
// operation, unless every load is proven in bounds and the body keeps to the
// fragment-safe contract.

gpu.module @kernels {
  gpu.func @constant_index(%in: memref<16xf32>, %out: memref<16xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 16, 1, 1>, vrc.profile = "gather-f32", vrc.synthesized_kernel, vrc.words = 16 : i32} {
    %x = gpu.global_id x
    %far = arith.constant 16 : index
    // expected-error @below {{synthesized gather kernel: load index for argument 0 is not proven in bounds: inferred [16, 16] but capacity is 16}}
    %value = memref.load %in[%far] : memref<16xf32>
    memref.store %value, %out[%x] : memref<16xf32>
    gpu.return
  }
}

// -----

gpu.module @kernels {
  gpu.func @fragment_index(%in: memref<16xf32>, %out: memref<64xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 64, 1, 1>, vrc.profile = "gather-f32", vrc.synthesized_kernel, vrc.words = 64 : i32} {
    %x = gpu.global_id x
    // expected-error @below {{load index for argument 0 is not proven in bounds: inferred [0, 63] but capacity is 16}}
    %value = memref.load %in[%x] : memref<16xf32>
    memref.store %value, %out[%x] : memref<64xf32>
    gpu.return
  }
}

// -----

// A scalar argument without vrc.argument_bounds may hold any value.
gpu.module @kernels {
  gpu.func @unbounded_scalar(%in: memref<16xf32>, %out: memref<16xf32>, %at: i32) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 16, 1, 1>, vrc.profile = "gather-f32", vrc.synthesized_kernel, vrc.words = 16 : i32} {
    %x = gpu.global_id x
    %index = arith.index_cast %at : i32 to index
    // expected-error @below {{load index for argument 0 has no proven range; capacity is 16}}
    %value = memref.load %in[%index] : memref<16xf32>
    memref.store %value, %out[%x] : memref<16xf32>
    gpu.return
  }
}

// -----

// The guard holds in the then region only.
gpu.module @kernels {
  gpu.func @unguarded_region(%in: memref<16xf32>, %out: memref<16xf32>, %at: i32) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 16, 1, 1>, vrc.profile = "gather-f32", vrc.synthesized_kernel, vrc.words = 16 : i32} {
    %x = gpu.global_id x
    %c16 = arith.constant 16 : i32
    %index = arith.index_cast %at : i32 to index
    %inside = arith.cmpi ult, %at, %c16 : i32
    %value = scf.if %inside -> (f32) {
      %zero = arith.constant 0.0 : f32
      scf.yield %zero : f32
    } else {
      // expected-error @below {{load index for argument 0 has no proven range; capacity is 16}}
      %loaded = memref.load %in[%index] : memref<16xf32>
      scf.yield %loaded : f32
    }
    memref.store %value, %out[%x] : memref<16xf32>
    gpu.return
  }
}

// -----

// A signed comparison does not exclude negative values.
gpu.module @kernels {
  gpu.func @signed_guard(%in: memref<16xf32>, %out: memref<16xf32>, %at: i32) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 16, 1, 1>, vrc.profile = "gather-f32", vrc.synthesized_kernel, vrc.words = 16 : i32} {
    %x = gpu.global_id x
    %c16 = arith.constant 16 : i32
    %index = arith.index_cast %at : i32 to index
    %inside = arith.cmpi slt, %at, %c16 : i32
    %value = scf.if %inside -> (f32) {
      // expected-error @below {{load index for argument 0 has no proven range; capacity is 16}}
      %loaded = memref.load %in[%index] : memref<16xf32>
      scf.yield %loaded : f32
    } else {
      %zero = arith.constant 0.0 : f32
      scf.yield %zero : f32
    }
    memref.store %value, %out[%x] : memref<16xf32>
    gpu.return
  }
}

// -----

// x - 1 wraps to 0xffffffff for the first fragment, so the masked sum is not
// the small value interval arithmetic without wrapping would report.
gpu.module @kernels {
  gpu.func @unsigned_wrap(%in: memref<16xf32>, %out: memref<16xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 16, 1, 1>, vrc.profile = "gather-f32", vrc.synthesized_kernel, vrc.words = 16 : i32} {
    %x = gpu.global_id x
    %i = arith.index_cast %x : index to i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %before = arith.subi %i, %c1 : i32
    %half = arith.divui %before, %c2 : i32
    %index = arith.index_cast %half : i32 to index
    // expected-error @below {{load index for argument 0 has no proven range; capacity is 16}}
    %value = memref.load %in[%index] : memref<16xf32>
    memref.store %value, %out[%x] : memref<16xf32>
    gpu.return
  }
}

// -----

// The product of two bounded scalars may exceed 32 bits.
gpu.module @kernels {
  gpu.func @multiply_wrap(%in: memref<16xf32>, %out: memref<16xf32>, %a: i32, %b: i32) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 16, 1, 1>, vrc.argument_bounds = [{index = 2 : i32, max = 65536 : i32, min = 0 : i32}, {index = 3 : i32, max = 65536 : i32, min = 0 : i32}], vrc.profile = "gather-f32", vrc.synthesized_kernel, vrc.words = 16 : i32} {
    %x = gpu.global_id x
    %c16 = arith.constant 16 : i32
    %product = arith.muli %a, %b : i32
    %shifted = arith.shrui %product, %c16 : i32
    %c28 = arith.constant 28 : i32
    %top = arith.shrui %shifted, %c28 : i32
    %index = arith.index_cast %top : i32 to index
    // expected-error @below {{load index for argument 0 has no proven range; capacity is 16}}
    %value = memref.load %in[%index] : memref<16xf32>
    memref.store %value, %out[%x] : memref<16xf32>
    gpu.return
  }
}

// -----

gpu.module @kernels {
  gpu.func @two_stores(%in: memref<16xf32>, %out: memref<16xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 16, 1, 1>, vrc.profile = "gather-f32", vrc.synthesized_kernel, vrc.words = 16 : i32} {
    %x = gpu.global_id x
    %value = memref.load %in[%x] : memref<16xf32>
    memref.store %value, %out[%x] : memref<16xf32>
    // expected-error @below {{synthesized gather kernel: requires exactly one output store}}
    memref.store %value, %out[%x] : memref<16xf32>
    gpu.return
  }
}

// -----

gpu.module @kernels {
  // expected-error @below {{synthesized gather kernel: requires exactly one output store}}
  gpu.func @no_store(%in: memref<16xf32>, %out: memref<16xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 16, 1, 1>, vrc.profile = "gather-f32", vrc.synthesized_kernel, vrc.words = 16 : i32} {
    gpu.return
  }
}

// -----

gpu.module @kernels {
  gpu.func @conditional_store(%in: memref<16xf32>, %out: memref<16xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 16, 1, 1>, vrc.profile = "gather-f32", vrc.synthesized_kernel, vrc.words = 16 : i32} {
    %x = gpu.global_id x
    %value = memref.load %in[%x] : memref<16xf32>
    %always = arith.constant true
    scf.if %always {
      // expected-error @below {{the store must write the output buffer at the top level of the kernel}}
      memref.store %value, %out[%x] : memref<16xf32>
    }
    gpu.return
  }
}

// -----

// The row term must scale by the grid width, not by another constant.
gpu.module @kernels {
  gpu.func @nonlinear_store(%in: memref<64xf32>, %out: memref<64xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 16, 4, 1>, vrc.profile = "gather-f32", vrc.synthesized_kernel, vrc.words = 64 : i32} {
    %x = gpu.global_id x
    %y = gpu.global_id y
    %value = memref.load %in[%x] : memref<64xf32>
    %pitch = arith.constant 8 : index
    %offset = arith.muli %y, %pitch : index
    %linear = arith.addi %x, %offset : index
    // expected-error @below {{the store index must be global_id x + global_id y * 16}}
    memref.store %value, %out[%linear] : memref<64xf32>
    gpu.return
  }
}

// -----

gpu.module @kernels {
  gpu.func @constant_store(%in: memref<16xf32>, %out: memref<16xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 16, 1, 1>, vrc.profile = "gather-f32", vrc.synthesized_kernel, vrc.words = 16 : i32} {
    %x = gpu.global_id x
    %value = memref.load %in[%x] : memref<16xf32>
    %first = arith.constant 0 : index
    // expected-error @below {{the store index must be global_id x + global_id y * 16}}
    memref.store %value, %out[%first] : memref<16xf32>
    gpu.return
  }
}

// -----

gpu.module @kernels {
  gpu.func @output_read(%in: memref<16xf32>, %out: memref<16xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 16, 1, 1>, vrc.profile = "gather-f32", vrc.synthesized_kernel, vrc.words = 16 : i32} {
    %x = gpu.global_id x
    // expected-error @below {{loads must read an input buffer argument}}
    %value = memref.load %out[%x] : memref<16xf32>
    memref.store %value, %out[%x] : memref<16xf32>
    gpu.return
  }
}

// -----

gpu.module @kernels {
  gpu.func @disallowed_operation(%in: memref<16xf32>, %out: memref<16xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 16, 1, 1>, vrc.profile = "gather-f32", vrc.synthesized_kernel, vrc.words = 16 : i32} {
    %x = gpu.global_id x
    %value = memref.load %in[%x] : memref<16xf32>
    // expected-error @below {{synthesized gather kernel: operation is not allowed: math.tanh}}
    %squashed = math.tanh %value : f32
    memref.store %squashed, %out[%x] : memref<16xf32>
    gpu.return
  }
}

// -----

gpu.module @kernels {
  gpu.func @fused_multiply_add(%in: memref<16xf32>, %out: memref<16xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 16, 1, 1>, vrc.profile = "gather-f32", vrc.synthesized_kernel, vrc.words = 16 : i32} {
    %x = gpu.global_id x
    %value = memref.load %in[%x] : memref<16xf32>
    // expected-error @below {{operation is not allowed: math.fma}}
    %fused = math.fma %value, %value, %value : f32
    memref.store %fused, %out[%x] : memref<16xf32>
    gpu.return
  }
}

// -----

gpu.module @kernels {
  gpu.func @fast_math(%in: memref<16xf32>, %out: memref<16xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 16, 1, 1>, vrc.profile = "gather-f32", vrc.synthesized_kernel, vrc.words = 16 : i32} {
    %x = gpu.global_id x
    %value = memref.load %in[%x] : memref<16xf32>
    // expected-error @below {{fast-math flags are not allowed}}
    %double = arith.addf %value, %value fastmath<contract> : f32
    memref.store %double, %out[%x] : memref<16xf32>
    gpu.return
  }
}

// -----

gpu.module @kernels {
  gpu.func @counted_for(%in: memref<16xf32>, %out: memref<16xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 16, 1, 1>, vrc.profile = "gather-f32", vrc.synthesized_kernel, vrc.words = 16 : i32} {
    %x = gpu.global_id x
    %first = arith.constant 0 : index
    %last = arith.constant 16 : index
    %step = arith.constant 1 : index
    %zero = arith.constant 0.0 : f32
    // expected-error @below {{operation is not allowed: scf.for}}
    %total = scf.for %i = %first to %last step %step iter_args(%sum = %zero) -> (f32) {
      %term = memref.load %in[%i] : memref<16xf32>
      %next = arith.addf %sum, %term : f32
      scf.yield %next : f32
    }
    memref.store %total, %out[%x] : memref<16xf32>
    gpu.return
  }
}

// -----

// A loop whose exit does not follow from a counted induction is not proven
// finite.
gpu.module @kernels {
  gpu.func @unbounded_loop(%in: memref<16xf32>, %out: memref<16xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 16, 1, 1>, vrc.profile = "gather-f32", vrc.synthesized_kernel, vrc.words = 16 : i32} {
    %x = gpu.global_id x
    %start = memref.load %in[%x] : memref<16xf32>
    %limit = arith.constant 1.0 : f32
    // expected-error @below {{loop must count a carried value up by one below a proven bound of at most 131072}}
    %total = scf.while (%sum = %start) : (f32) -> (f32) {
      %more = arith.cmpf olt, %sum, %limit : f32
      scf.condition(%more) %sum : f32
    } do {
    ^bb0(%partial: f32):
      %next = arith.addf %partial, %partial : f32
      scf.yield %next : f32
    }
    memref.store %total, %out[%x] : memref<16xf32>
    gpu.return
  }
}

// -----

gpu.module @kernels {
  // expected-error @below {{synthesized gather kernel: vrc.words must equal width * height and be at most 131072}}
  gpu.func @words_mismatch(%in: memref<16xf32>, %out: memref<16xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 16, 2, 1>, vrc.profile = "gather-f32", vrc.synthesized_kernel, vrc.words = 16 : i32} {
    %x = gpu.global_id x
    %value = memref.load %in[%x] : memref<16xf32>
    memref.store %value, %out[%x] : memref<16xf32>
    gpu.return
  }
}

// -----

gpu.module @kernels {
  // expected-error @below {{synthesized gather kernel: output argument 1 must hold exactly vrc.words = 16 elements}}
  gpu.func @output_size(%in: memref<16xf32>, %out: memref<32xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 16, 1, 1>, vrc.profile = "gather-f32", vrc.synthesized_kernel, vrc.words = 16 : i32} {
    %x = gpu.global_id x
    %value = memref.load %in[%x] : memref<16xf32>
    memref.store %value, %out[%x] : memref<32xf32>
    gpu.return
  }
}

// -----

gpu.module @kernels {
  // expected-error @below {{synthesized gather kernel: argument 2 must be a static one-dimensional f32/i32 buffer that precedes every scalar}}
  gpu.func @scalar_before_buffer(%in: memref<16xf32>, %scale: f32, %out: memref<16xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 16, 1, 1>, vrc.profile = "gather-f32", vrc.synthesized_kernel, vrc.words = 16 : i32} {
    %x = gpu.global_id x
    %value = memref.load %in[%x] : memref<16xf32>
    memref.store %value, %out[%x] : memref<16xf32>
    gpu.return
  }
}

// -----

gpu.module @kernels {
  // expected-error @below {{synthesized gather kernel: vrc.argument_bounds must uniquely describe i32 scalar arguments as {index, min, max}}}
  gpu.func @float_bound(%in: memref<16xf32>, %out: memref<16xf32>, %scale: f32) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 16, 1, 1>, vrc.argument_bounds = [{index = 2 : i32, max = 4 : i32, min = 0 : i32}], vrc.profile = "gather-f32", vrc.synthesized_kernel, vrc.words = 16 : i32} {
    %x = gpu.global_id x
    %value = memref.load %in[%x] : memref<16xf32>
    memref.store %value, %out[%x] : memref<16xf32>
    gpu.return
  }
}
