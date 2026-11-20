// RUN: %vrc-opt --vrc-prove-synthesized-gather-kernels --vrc-lower-gather-device %s | %FileCheck %s

// A proven synthesized kernel is accepted by the unchanged gather lowering
// and by the vrcgpu.kernel verifier: buffers become point-loaded textures and
// the single store becomes the packed fragment output.

// CHECK: module attributes {vrc.device_stage = "vrcgpu-gather-f32"}
// CHECK: "vrcgpu.kernel"()
// CHECK-SAME: argument_kinds = ["buffer", "buffer", "buffer", "f32", "i32"]
// CHECK-SAME: entry_name = "scaled_row"
// CHECK-SAME: global_size = array<i32: 64, 4, 1>
// CHECK-SAME: kernel_id = 7
// CHECK-SAME: sym_name = "synthesized__scaled_row"
// CHECK-NEXT: ^bb0(%[[FULL:.*]]: !vrcgpu.version<0>, %[[ROW:.*]]: !vrcgpu.version<1>):
// CHECK-NEXT: "vrcgpu.resource"() <{arg_index = 0 : i32, {{.*}}kind = "texture"
// CHECK-NEXT: "vrcgpu.resource"() <{arg_index = 1 : i32, {{.*}}kind = "texture"
// CHECK-NEXT: "vrcgpu.resource"() <{arg_index = 2 : i32, {{.*}}kind = "target"
// CHECK-NEXT: "vrcgpu.resource"() <{arg_index = 3 : i32, {{.*}}kind = "uniform"
// CHECK-NEXT: "vrcgpu.resource"() <{arg_index = 4 : i32, {{.*}}kind = "uniform"
// CHECK-NEXT: "vrcgpu.pass"(%[[FULL]], %[[ROW]]) <{output = @arg2, pass_id = 0 : i32}>
// CHECK-DAG: %[[SCALE:.*]] = "vrcgpu.uniform_load"() <{resource = @arg3}> : () -> f32
// CHECK-DAG: %[[COUNT:.*]] = "vrcgpu.uniform_load"() <{resource = @arg4}> : () -> i32
// CHECK: "vrcgpu.fragment_coord"()
// CHECK: arith.cmpi ult, %{{.*}}, %[[COUNT]] : i32
// CHECK: %[[VALUE:.*]] = scf.if
// The full-size input shares the output texture geometry.
// CHECK: %[[TEXEL0:.*]] = "vrcgpu.texture_load"(%{{.*}}, %{{.*}}) <{element_count = 256 : i32, resource = @arg0}>
// CHECK: %[[A:.*]] = "vrcgpu.raw_unpack"(%[[TEXEL0]])
// The 64-element input is one texture row of its own width.
// CHECK: %[[WIDTH:.*]] = arith.constant 64 : i32
// CHECK: %[[Y:.*]] = arith.divui %[[COLUMN:.*]], %[[WIDTH]] : i32
// CHECK: %[[X:.*]] = arith.remui %[[COLUMN]], %[[WIDTH]] : i32
// CHECK: %[[TEXEL1:.*]] = "vrcgpu.texture_load"(%[[X]], %[[Y]]) <{element_count = 64 : i32, resource = @arg1}>
// CHECK: %[[B:.*]] = "vrcgpu.raw_unpack"(%[[TEXEL1]])
// CHECK: %[[PRODUCT:.*]] = arith.mulf %[[A]], %[[B]] : f32
// CHECK: arith.mulf %[[PRODUCT]], %[[SCALE]] : f32
// CHECK: %[[PACKED:.*]] = "vrcgpu.raw_pack"(%[[VALUE]])
// CHECK: "vrcgpu.output"(%[[PACKED]])
// CHECK: vrc.argument_bounds = [{index = 4 : i32, max = 256 : i32, min = 0 : i32}]
// CHECK-SAME: vrc.gather_proven
// CHECK-SAME: vrc.profile = "gather-f32"
// CHECK-SAME: vrc.texture_shape = array<i32: 64, 4>
// CHECK-SAME: vrc.words = 256
// CHECK-NOT: gpu.module

gpu.module @synthesized {
  gpu.func @scaled_row(%full: memref<256xf32>, %row: memref<64xf32>, %out: memref<256xf32>, %scale: f32, %count: i32) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 64, 4, 1>, vrc.argument_bounds = [{index = 4 : i32, max = 256 : i32, min = 0 : i32}], vrc.kernel_id = 7 : i32, vrc.profile = "gather-f32", vrc.synthesized_kernel, vrc.words = 256 : i32} {
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
      %product = arith.mulf %a, %b : f32
      %scaled = arith.mulf %product, %scale : f32
      scf.yield %scaled : f32
    } else {
      %zero = arith.constant 0.0 : f32
      scf.yield %zero : f32
    }
    memref.store %value, %out[%linear] : memref<256xf32>
    gpu.return
  }
}
