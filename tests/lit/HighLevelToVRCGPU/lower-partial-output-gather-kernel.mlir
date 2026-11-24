// RUN: %vrc-opt --split-input-file --vrc-prove-synthesized-gather-kernels --vrc-lower-gather-device %s | %FileCheck %s

// A partial-output kernel stores its word only where a condition holds: the
// store ends a top-level scf.if. The pass lowers the condition to the
// `keep` operand of the output, and the fragments where it is false write
// nothing (HLSL discard), so their texels keep their contents. The stored
// word is computed under the condition (zero elsewhere), and only the store
// is conditional.

// CHECK-LABEL: entry_name = "rows"
// CHECK: %[[KEEP:.*]] = arith.andi
// CHECK: %[[VALUE:.*]] = scf.if %[[KEEP]] -> (f32)
// CHECK: %[[KEPT:.*]] = scf.if %[[KEEP]] -> (f32) {
// CHECK-NEXT: scf.yield %[[VALUE]] : f32
// CHECK: %[[TEXEL:.*]] = "vrcgpu.raw_pack"(%[[KEPT]])
// CHECK-NOT: memref.store
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

// -----

// Canonicalization merges the branch that computes the word into the store
// branch of the same condition. The pass computes the word under that
// condition again (zero elsewhere) and outputs it with the condition.

// CHECK-LABEL: entry_name = "merged_rows"
// CHECK: %[[KEEP:.*]] = arith.cmpi uge
// CHECK: %[[VALUE:.*]] = scf.if %[[KEEP]] -> (f32) {
// CHECK: %[[SAMPLE:.*]] = "vrcgpu.raw_unpack"
// CHECK: %[[TWICE:.*]] = arith.addf %[[SAMPLE]], %[[SAMPLE]]
// CHECK: scf.yield %[[TWICE]] : f32
// CHECK: } else {
// CHECK: scf.yield %{{.*}} : f32
// CHECK: %[[TEXEL:.*]] = "vrcgpu.raw_pack"(%[[VALUE]])
// CHECK: "vrcgpu.output"(%[[TEXEL]], %[[KEEP]])

gpu.module @partial {
  gpu.func @merged_rows(%input: memref<64xf32>, %out: memref<64xf32>, %first: i32) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 64, 1, 1>, vrc.argument_bounds = [{index = 2 : i64, max = 4 : i64, min = 0 : i64}], vrc.kernel_id = 7 : i32, vrc.output_granule = 16 : i64, vrc.profile = "gather-f32", vrc.synthesized_kernel, vrc.words = 64 : i32} {
    %x = gpu.global_id x
    %i = arith.index_cast %x : index to i32
    %c16 = arith.constant 16 : i32
    %c4 = arith.constant 4 : i32
    %row = arith.divui %i, %c16 : i32
    %bounded = arith.minui %first, %c4 : i32
    %keep = arith.cmpi uge, %row, %bounded : i32
    scf.if %keep {
      %sample = memref.load %input[%x] : memref<64xf32>
      %twice = arith.addf %sample, %sample : f32
      memref.store %twice, %out[%x] : memref<64xf32>
    }
    gpu.return
  }
}
