// RUN: %vrc-opt --split-input-file --vrc-prove-synthesized-gather-kernels --vrc-lower-gather-device %s | %FileCheck %s

// A partial-output kernel stores its word only where a condition holds: the
// store is alone in a top-level scf.if. The pass lowers the condition to the
// `keep` operand of the output, and the fragments where it is false write
// nothing (HLSL discard), so their texels keep their contents. The value is
// computed as for any other kernel; only the store is conditional.

// CHECK-LABEL: entry_name = "rows"
// CHECK: %[[KEEP:.*]] = arith.andi
// CHECK: %[[VALUE:.*]] = scf.if %[[KEEP]] -> (f32)
// CHECK: %[[TEXEL:.*]] = "vrcgpu.raw_pack"(%[[VALUE]])
// CHECK-NOT: scf.if %[[KEEP]] {
// CHECK: "vrcgpu.output"(%[[TEXEL]], %[[KEEP]])
// CHECK: vrc.gather_proven

gpu.module @partial {
  gpu.func @rows(%input: memref<64xf32>, %out: memref<64xf32>, %first: i32) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 64, 1, 1>, vrc.argument_bounds = [{index = 2 : i64, max = 4 : i64, min = 0 : i64}], vrc.kernel_id = 5 : i32, vrc.output_granule = 16 : i64, vrc.profile = "gather-f32", vrc.synthesized_kernel, vrc.words = 64 : i32} {
    %x = gpu.global_id x
    %i = arith.index_cast %x : index to i32
    %c16 = arith.constant 16 : i32
    %c4 = arith.constant 4 : i32
    %row = arith.divui %i, %c16 : i32
    %bounded = arith.minui %first, %c4 : i32
    %after = arith.cmpi uge, %row, %bounded : i32
    %live = arith.cmpi ult, %i, %c16 : i32
    %keep = arith.andi %after, %live : i1
    %value = scf.if %keep -> (f32) {
      %sample = memref.load %input[%x] : memref<64xf32>
      scf.yield %sample : f32
    } else {
      %zero = arith.constant 0.0 : f32
      scf.yield %zero : f32
    }
    scf.if %keep {
      memref.store %value, %out[%x] : memref<64xf32>
    }
    gpu.return
  }
}

// -----

// With four words per texel, the output word carries the condition too; the
// granule of 16 words is whole texels.

// CHECK-LABEL: entry_name = "word_rows"
// CHECK: %[[KEEP:.*]] = arith.cmpi uge
// CHECK: vrcgpu.output_word %{{.*}} if %[[KEEP]] : f32
// CHECK: vrc.gather_proven
// CHECK-SAME: vrc.texel_format = "rgba32_float"

gpu.module @partial {
  gpu.func @word_rows(%input: memref<64xf32>, %out: memref<64xf32>, %first: i32) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 64, 1, 1>, vrc.argument_bounds = [{index = 2 : i64, max = 4 : i64, min = 0 : i64}], vrc.kernel_id = 6 : i32, vrc.output_granule = 16 : i64, vrc.profile = "gather-f32", vrc.synthesized_kernel, vrc.texel_format = "rgba32_float", vrc.words = 64 : i32} {
    %x = gpu.global_id x
    %i = arith.index_cast %x : index to i32
    %c16 = arith.constant 16 : i32
    %c4 = arith.constant 4 : i32
    %row = arith.divui %i, %c16 : i32
    %bounded = arith.minui %first, %c4 : i32
    %keep = arith.cmpi uge, %row, %bounded : i32
    %sample = memref.load %input[%x] : memref<64xf32>
    scf.if %keep {
      memref.store %sample, %out[%x] : memref<64xf32>
    }
    gpu.return
  }
}
