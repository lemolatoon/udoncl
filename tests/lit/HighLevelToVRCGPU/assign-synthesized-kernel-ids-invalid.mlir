// RUN: %vrc-opt --split-input-file --verify-diagnostics --vrc-assign-synthesized-kernel-ids %s

// A dispatch declaration must name a synthesized kernel of the module and
// repeat its argument list with an i32 token for every buffer.

// expected-error @below {{vrc.synthesized_dispatch must be a symbol reference @gpu_module::@kernel}}
func.func private @flat_reference(i32, i32) -> i32 attributes {vrc.synthesized_dispatch = @kernel}

// -----

// expected-error @below {{vrc.synthesized_dispatch @carrier::@missing does not name a synthesized kernel}}
func.func private @dangling(i32, i32) -> i32 attributes {vrc.synthesized_dispatch = @carrier::@missing}

// -----

module {
  gpu.module @library {
    gpu.func @kernel(%in: memref<4xf32>, %out: memref<4xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 4, 1, 1>, vrc.kernel_id = 5 : i32} {
      gpu.return
    }
  }
  // expected-error @below {{vrc.synthesized_dispatch @library::@kernel does not name a synthesized kernel}}
  func.func private @not_synthesized(i32, i32) -> i32 attributes {vrc.synthesized_dispatch = @library::@kernel}
}

// -----

module {
  // expected-error @below {{a gpu.module that carries synthesized kernels may only contain synthesized kernels}}
  gpu.module @mixed {
    gpu.func @kernel(%in: memref<4xf32>, %out: memref<4xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 4, 1, 1>, vrc.synthesized_kernel} {
      gpu.return
    }
    gpu.func @helper() {
      gpu.return
    }
  }
}

// -----

module {
  gpu.module @carrier {
    // expected-note @below {{kernel declared here}}
    gpu.func @kernel(%in: memref<4xf32>, %out: memref<4xf32>, %count: i32) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 4, 1, 1>, vrc.synthesized_kernel} {
      gpu.return
    }
  }
  // expected-error @below {{vrc.synthesized_dispatch declaration does not match kernel @kernel: expected 3 arguments but found 2}}
  func.func private @short(i32, i32) -> i32 attributes {vrc.synthesized_dispatch = @carrier::@kernel}
  func.func @host(%a: i32, %b: i32) -> i32 {
    %event = func.call @short(%a, %b) : (i32, i32) -> i32
    return %event : i32
  }
}

// -----

module {
  gpu.module @carrier {
    // expected-note @below {{kernel declared here}}
    gpu.func @kernel(%in: memref<4xf32>, %out: memref<4xf32>, %count: i32) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 4, 1, 1>, vrc.synthesized_kernel} {
      gpu.return
    }
  }
  // expected-error @below {{vrc.synthesized_dispatch declaration does not match kernel @kernel: argument 2 must be 'i32' but is 'f32'}}
  func.func private @wrong_scalar(i32, i32, f32) -> i32 attributes {vrc.synthesized_dispatch = @carrier::@kernel}
  func.func @host(%a: i32, %b: i32, %c: f32) -> i32 {
    %event = func.call @wrong_scalar(%a, %b, %c) : (i32, i32, f32) -> i32
    return %event : i32
  }
}

// -----

module {
  gpu.module @carrier {
    // expected-note @below {{kernel declared here}}
    gpu.func @kernel(%in: memref<4xf32>, %out: memref<4xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 4, 1, 1>, vrc.synthesized_kernel} {
      gpu.return
    }
  }
  // expected-error @below {{vrc.synthesized_dispatch declaration does not match kernel @kernel: argument 1 must be 'i32' but is 'memref<4xf32>'}}
  func.func private @memref_buffer(i32, memref<4xf32>) -> i32 attributes {vrc.synthesized_dispatch = @carrier::@kernel}
  func.func @host(%a: i32, %b: memref<4xf32>) -> i32 {
    %event = func.call @memref_buffer(%a, %b) : (i32, memref<4xf32>) -> i32
    return %event : i32
  }
}

// -----

module {
  gpu.module @carrier {
    // expected-note @below {{kernel declared here}}
    gpu.func @kernel(%in: memref<4xf32>, %out: memref<4xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 4, 1, 1>, vrc.synthesized_kernel} {
      gpu.return
    }
  }
  // expected-error @below {{vrc.synthesized_dispatch declaration does not match kernel @kernel: requires an external declaration returning i32}}
  func.func private @no_status(i32, i32) attributes {vrc.synthesized_dispatch = @carrier::@kernel}
  func.func @host(%a: i32, %b: i32) {
    func.call @no_status(%a, %b) : (i32, i32) -> ()
    return
  }
}

// -----

module {
  gpu.module @carrier {
    // expected-error @below {{synthesized kernel: vrc.kernel_id is assigned by the compiler}}
    gpu.func @kernel(%in: memref<4xf32>, %out: memref<4xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 4, 1, 1>, vrc.kernel_id = 3 : i32, vrc.synthesized_kernel} {
      gpu.return
    }
  }
  func.func private @dispatch(i32, i32) -> i32 attributes {vrc.synthesized_dispatch = @carrier::@kernel}
  func.func @host(%a: i32, %b: i32) -> i32 {
    %event = func.call @dispatch(%a, %b) : (i32, i32) -> i32
    return %event : i32
  }
}

// -----

module {
  gpu.module @carrier {
    // expected-error @below {{synthesized kernel: requires a gpu.kernel with a three-dimensional known_grid_size}}
    gpu.func @kernel(%in: memref<4xf32>, %out: memref<4xf32>) kernel attributes {vrc.synthesized_kernel} {
      gpu.return
    }
  }
  func.func private @dispatch(i32, i32) -> i32 attributes {vrc.synthesized_dispatch = @carrier::@kernel}
  func.func @host(%a: i32, %b: i32) -> i32 {
    %event = func.call @dispatch(%a, %b) : (i32, i32) -> i32
    return %event : i32
  }
}

// -----

module {
  gpu.module @carrier {
    // expected-error @below {{synthesized kernel: requires at least one buffer argument, and every buffer before the first scalar}}
    gpu.func @kernel(%in: memref<4xf32>, %count: i32, %out: memref<4xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 4, 1, 1>, vrc.synthesized_kernel} {
      gpu.return
    }
  }
  func.func private @dispatch(i32, i32, i32) -> i32 attributes {vrc.synthesized_dispatch = @carrier::@kernel}
  func.func @host(%a: i32, %b: i32) -> i32 {
    %event = func.call @dispatch(%a, %b, %a) : (i32, i32, i32) -> i32
    return %event : i32
  }
}

// -----

// expected-error @below {{symbol @device_gpu_1 is reserved for synthesized kernel 1}}
module {
  gpu.module @device_gpu_1 {
  }
  gpu.module @carrier {
    gpu.func @kernel(%in: memref<4xf32>, %out: memref<4xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 4, 1, 1>, vrc.synthesized_kernel} {
      gpu.return
    }
  }
  func.func private @dispatch(i32, i32) -> i32 attributes {vrc.synthesized_dispatch = @carrier::@kernel}
  func.func @host(%a: i32, %b: i32) -> i32 {
    %event = func.call @dispatch(%a, %b) : (i32, i32) -> i32
    return %event : i32
  }
}
