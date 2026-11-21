// RUN: %vrc-opt --vrc-prove-synthesized-gather-kernels --vrc-lower-gather-device %s | %FileCheck %s

// With four words per texel (rgba32_float) the pass keeps one-word semantics:
// every input read is a word_load by its linear word index and row pitch,
// the stored value is the output word, and every buffer resource and the
// kernel name the texel format. Equal index expressions are merged so the
// emitter sees the loads that share texels.

// CHECK: "vrcgpu.kernel"()
// CHECK-SAME: entry_name = "scaled_row"
// CHECK-NEXT: ^bb0(%[[FULL:.*]]: !vrcgpu.version<0>, %[[ROW:.*]]: !vrcgpu.version<1>):
// CHECK-NEXT: "vrcgpu.resource"() <{arg_index = 0 : i32, {{.*}}format = "rgba32_float", kind = "texture"
// CHECK-NEXT: "vrcgpu.resource"() <{arg_index = 1 : i32, {{.*}}format = "rgba32_float", kind = "texture"
// CHECK-NEXT: "vrcgpu.resource"() <{arg_index = 2 : i32, {{.*}}format = "rgba32_float", kind = "target"
// CHECK: "vrcgpu.fragment_coord"()
// CHECK: %[[VALUE:.*]] = scf.if
// CHECK: vrcgpu.word_load @arg0[%{{.*}}] {element_count = 256 : i32, row_pitch = 64 : i32} : f32
// The 64-element input is one row of its own width.
// CHECK: vrcgpu.word_load @arg1[%{{.*}}] {element_count = 64 : i32, row_pitch = 64 : i32} : f32
// CHECK-NOT: vrcgpu.word_load
// CHECK: vrcgpu.output_word %[[VALUE]] : f32
// CHECK: vrc.gather_proven
// CHECK-SAME: vrc.texel_format = "rgba32_float"
// CHECK-SAME: vrc.texture_shape = array<i32: 64, 4>
// CHECK-NOT: vrcgpu.texture_load
// CHECK-NOT: vrcgpu.raw_pack

gpu.module @synthesized {
  gpu.func @scaled_row(%full: memref<256xf32>, %row: memref<64xf32>, %out: memref<256xf32>, %scale: f32, %count: i32) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 64, 4, 1>, vrc.argument_bounds = [{index = 4 : i32, max = 256 : i32, min = 0 : i32}], vrc.kernel_id = 7 : i32, vrc.profile = "gather-f32", vrc.synthesized_kernel, vrc.texel_format = "rgba32_float", vrc.words = 256 : i32} {
    %x = gpu.global_id x
    %y = gpu.global_id y
    %width = arith.constant 64 : index
    %offset = arith.muli %y, %width : index
    %linear = arith.addi %x, %offset : index
    %i = arith.index_cast %linear : index to i32
    %live = arith.cmpi ult, %i, %count : i32
    %value = scf.if %live -> (f32) {
      %c64 = arith.constant 64 : i32
      %column = arith.remui %i, %c64 : i32
      %column_index = arith.index_cast %column : i32 to index
      %a = memref.load %full[%linear] : memref<256xf32>
      %b = memref.load %row[%column_index] : memref<64xf32>
      %again = memref.load %row[%column_index] : memref<64xf32>
      %product = arith.mulf %a, %b : f32
      %scaled = arith.mulf %product, %again : f32
      scf.yield %scaled : f32
    } else {
      %zero = arith.constant 0.0 : f32
      scf.yield %zero : f32
    }
    memref.store %value, %out[%linear] : memref<256xf32>
    gpu.return
  }
}
