// Rotary position embedding of a [1, 4, 1, 64] tensor as one synthesized
// gather kernel:
//   out[i] = x[i] * cos[i % 64] + rot(x)[i] * sin[i % 64]
// where rot swaps the two halves of every 64-element row and negates the half
// it moves down. Fragments from `count` on write zero. The unfused form is
// three UdonDNN OpTensor dispatches plus the rotation.
gpu.module @synthesized {
  gpu.func @synthesized_rotary(%x: memref<256xf32>, %cos: memref<64xf32>, %sin: memref<64xf32>, %out: memref<131072xf32>, %count: i32) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 8192, 16, 1>, vrc.argument_bounds = [{index = 4 : i32, max = 256 : i32, min = 0 : i32}], vrc.profile = "gather-f32", vrc.synthesized_kernel, vrc.words = 131072 : i32} {
    %gx = gpu.global_id x
    %gy = gpu.global_id y
    %width = arith.constant 8192 : index
    %row = arith.muli %gy, %width : index
    %linear = arith.addi %gx, %row : index
    %i = arith.index_cast %linear : index to i32
    %live = arith.cmpi ult, %i, %count : i32
    %value = scf.if %live -> (f32) {
      %c32 = arith.constant 32 : i32
      %c64 = arith.constant 64 : i32
      %column = arith.remui %i, %c64 : i32
      %line = arith.divui %i, %c64 : i32
      %base = arith.muli %line, %c64 : i32
      %low = arith.cmpi ult, %column, %c32 : i32
      %partner = scf.if %low -> (i32) {
        %up = arith.addi %column, %c32 : i32
        scf.yield %up : i32
      } else {
        %down = arith.subi %column, %c32 : i32
        scf.yield %down : i32
      }
      %source = arith.addi %base, %partner : i32
      %source_index = arith.index_cast %source : i32 to index
      %column_index = arith.index_cast %column : i32 to index
      %direct = memref.load %x[%linear] : memref<256xf32>
      %moved = memref.load %x[%source_index] : memref<256xf32>
      %negated = arith.negf %moved : f32
      %rotated = arith.select %low, %negated, %moved : f32
      %c = memref.load %cos[%column_index] : memref<64xf32>
      %s = memref.load %sin[%column_index] : memref<64xf32>
      %kept = arith.mulf %direct, %c : f32
      %crossed = arith.mulf %rotated, %s : f32
      %sum = arith.addf %kept, %crossed : f32
      scf.yield %sum : f32
    } else {
      %zero = arith.constant 0.0 : f32
      scf.yield %zero : f32
    }
    memref.store %value, %out[%linear] : memref<131072xf32>
    gpu.return
  }
}
