// RUN: %vrc-opt --vrc-prove-synthesized-gather-kernels %s | %FileCheck %s

// A synthesized gather kernel is marked proven once every load index is shown
// to stay inside its input buffer; the marker that requested the proof stays.

gpu.module @kernels {
  // Broadcast-style index arithmetic: every index is bounded by the fragment
  // coordinates and constants alone.
  // CHECK-LABEL: gpu.func @broadcast
  // CHECK-SAME: vrc.gather_proven
  // CHECK-SAME: vrc.synthesized_kernel
  gpu.func @broadcast(%full: memref<256xf32>, %row: memref<64xf32>, %channel: memref<4xf32>, %out: memref<256xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 64, 4, 1>, vrc.profile = "gather-f32", vrc.synthesized_kernel, vrc.words = 256 : i32} {
    %x = gpu.global_id x
    %y = gpu.global_id y
    %width = arith.constant 64 : index
    %offset = arith.muli %y, %width : index
    %linear = arith.addi %x, %offset : index
    %i = arith.index_cast %linear : index to i32
    %c64 = arith.constant 64 : i32
    %column = arith.remui %i, %c64 : i32
    %line = arith.divui %i, %c64 : i32
    %base = arith.muli %line, %c64 : i32
    %c63 = arith.constant 63 : i32
    %mirror = arith.subi %c63, %column : i32
    %flipped = arith.addi %base, %mirror : i32
    %flipped_index = arith.index_cast %flipped : i32 to index
    %column_index = arith.index_cast %column : i32 to index
    %line_index = arith.index_cast %line : i32 to index
    %a = memref.load %full[%flipped_index] : memref<256xf32>
    %b = memref.load %row[%column_index] : memref<64xf32>
    %c = memref.load %channel[%line_index] : memref<4xf32>
    %product = arith.mulf %a, %b : f32
    %sum = arith.addf %product, %c : f32
    memref.store %sum, %out[%linear] : memref<256xf32>
    gpu.return
  }

  // An index is only bounded where a dominating scf.if condition guards it,
  // and the bound itself comes from vrc.argument_bounds.
  // CHECK-LABEL: gpu.func @guarded
  // CHECK-SAME: vrc.gather_proven
  gpu.func @guarded(%in: memref<100xf32>, %out: memref<8192xf32>, %count: i32) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 8192, 1, 1>, vrc.argument_bounds = [{index = 2 : i32, max = 100 : i32, min = 0 : i32}], vrc.profile = "gather-f32", vrc.synthesized_kernel, vrc.words = 8192 : i32} {
    %x = gpu.global_id x
    %i = arith.index_cast %x : index to i32
    %live = arith.cmpi ult, %i, %count : i32
    %value = scf.if %live -> (f32) {
      %loaded = memref.load %in[%x] : memref<100xf32>
      scf.yield %loaded : f32
    } else {
      %zero = arith.constant 0.0 : f32
      scf.yield %zero : f32
    }
    memref.store %value, %out[%x] : memref<8192xf32>
    gpu.return
  }

  // Index selection: a select of two bounded indices, a select whose
  // condition clamps an otherwise unbounded scalar, and scf.if results whose
  // yields are each bounded by the branch condition.
  // CHECK-LABEL: gpu.func @selected
  // CHECK-SAME: vrc.gather_proven
  gpu.func @selected(%in: memref<64xf32>, %out: memref<128xf32>, %start: i32) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 128, 1, 1>, vrc.profile = "gather-f32", vrc.synthesized_kernel, vrc.words = 128 : i32} {
    %x = gpu.global_id x
    %i = arith.index_cast %x : index to i32
    %c0 = arith.constant 0 : i32
    %c32 = arith.constant 32 : i32
    %c64 = arith.constant 64 : i32
    %j = arith.remui %i, %c64 : i32
    %low = arith.cmpi ult, %j, %c32 : i32
    %half = arith.divui %i, %c64 : i32
    %either = arith.select %low, %j, %half : i32
    %either_index = arith.index_cast %either : i32 to index
    %a = memref.load %in[%either_index] : memref<64xf32>
    %inside = arith.cmpi ult, %start, %c64 : i32
    %clamped = arith.select %inside, %start, %c0 : i32
    %clamped_index = arith.index_cast %clamped : i32 to index
    %b = memref.load %in[%clamped_index] : memref<64xf32>
    %rotated = scf.if %low -> (i32) {
      %up = arith.addi %j, %c32 : i32
      scf.yield %up : i32
    } else {
      %down = arith.subi %j, %c32 : i32
      scf.yield %down : i32
    }
    %rotated_index = arith.index_cast %rotated : i32 to index
    %c = memref.load %in[%rotated_index] : memref<64xf32>
    %ab = arith.addf %a, %b : f32
    %abc = arith.addf %ab, %c : f32
    memref.store %abc, %out[%x] : memref<128xf32>
    gpu.return
  }

  // A counted scf.while bounds its induction value inside the body.
  // CHECK-LABEL: gpu.func @reduced
  // CHECK-SAME: vrc.gather_proven
  gpu.func @reduced(%in: memref<1024xf32>, %out: memref<64xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 64, 1, 1>, vrc.profile = "gather-f32", vrc.synthesized_kernel, vrc.words = 64 : i32} {
    %x = gpu.global_id x
    %i = arith.index_cast %x : index to i32
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : i32
    %zero = arith.constant 0.0 : f32
    %base = arith.muli %i, %c16 : i32
    %k, %total = scf.while (%n = %c0, %sum = %zero) : (i32, f32) -> (i32, f32) {
      %more = arith.cmpi ult, %n, %c16 : i32
      scf.condition(%more) %n, %sum : i32, f32
    } do {
    ^bb0(%m: i32, %partial: f32):
      %at = arith.addi %base, %m : i32
      %at_index = arith.index_cast %at : i32 to index
      %term = memref.load %in[%at_index] : memref<1024xf32>
      %next_sum = arith.addf %partial, %term : f32
      %next = arith.addi %m, %c1 : i32
      scf.yield %next, %next_sum : i32, f32
    }
    memref.store %total, %out[%x] : memref<64xf32>
    gpu.return
  }

  // Kernels the compiler did not synthesize are not this pass's concern.
  // CHECK-LABEL: gpu.func @untouched
  // CHECK-NOT: vrc.gather_proven
  // CHECK: gpu.return
  gpu.func @untouched(%in: memref<4xf32>, %out: memref<4xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 4, 1, 1>, vrc.profile = "gather-f32", vrc.words = 4 : i32} {
    %far = arith.constant 9 : index
    %value = memref.load %in[%far] : memref<4xf32>
    %x = gpu.global_id x
    memref.store %value, %out[%x] : memref<4xf32>
    gpu.return
  }

  // A launch surface may be taller than 16 rows: up to a whole Device buffer
  // texture of 8192 x 8192 words. This copy writes 8192 x 128 words.
  // CHECK-LABEL: gpu.func @tall_surface
  // CHECK-SAME: vrc.gather_proven
  gpu.func @tall_surface(%in: memref<1048576xf32>, %out: memref<1048576xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 8192, 128, 1>, vrc.profile = "gather-f32", vrc.synthesized_kernel, vrc.words = 1048576 : i32} {
    %x = gpu.global_id x
    %y = gpu.global_id y
    %width = arith.constant 8192 : index
    %offset = arith.muli %y, %width : index
    %linear = arith.addi %x, %offset : index
    %value = memref.load %in[%linear] : memref<1048576xf32>
    memref.store %value, %out[%linear] : memref<1048576xf32>
    gpu.return
  }
}
