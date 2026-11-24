// A synthesized partial-output kernel (word_texel.py): rows of 64 words in
// [%first, %first + %count) of the result are twice the input, and every
// other word keeps the contents the target had before the draw.
gpu.module @partial {
  gpu.func @partial_rows(%input: memref<16384xf32>, %out: memref<16384xf32>, %first: i32, %count: i32) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 8192, 2, 1>, vrc.argument_bounds = [{index = 2 : i64, max = 256 : i64, min = 0 : i64}, {index = 3 : i64, max = 256 : i64, min = 0 : i64}], vrc.kernel_id = 9 : i32, vrc.output_granule = 64 : i64, vrc.profile = "gather-f32", vrc.synthesized_kernel, vrc.words = 16384 : i32} {
    %x = gpu.global_id x
    %y = gpu.global_id y
    %width = arith.constant 8192 : index
    %offset = arith.muli %y, %width : index
    %linear = arith.addi %x, %offset : index
    %i = arith.index_cast %linear : index to i32
    %c64 = arith.constant 64 : i32
    %c256 = arith.constant 256 : i32
    %first_bounded = arith.minui %first, %c256 : i32
    %count_bounded = arith.minui %count, %c256 : i32
    %row = arith.divui %i, %c64 : i32
    %end = arith.addi %first_bounded, %count_bounded : i32
    %after = arith.cmpi uge, %row, %first_bounded : i32
    %before = arith.cmpi ult, %row, %end : i32
    %keep = arith.andi %after, %before : i1
    %value = scf.if %keep -> (f32) {
      %sample = memref.load %input[%linear] : memref<16384xf32>
      %twice = arith.addf %sample, %sample : f32
      scf.yield %twice : f32
    } else {
      %zero = arith.constant 0.0 : f32
      scf.yield %zero : f32
    }
    scf.if %keep {
      memref.store %value, %out[%linear] : memref<16384xf32>
    }
    gpu.return
  }
}
