// Host half of the generated kernel channel test. unity_synthesized_dispatch.py
// appends it to synthesized_rotary.mlir, whose gpu.module @synthesized carries
// the kernel that @__vrc_opencl_dispatch_rotary names. The Host scales a
// tensor with the configured UdonDNN OpTensor library kernel and then rotates
// the result with the synthesized kernel.
func.func private @__vrc_opencl_buffer_alloc(i32, i32) -> i32
func.func private @__vrc_opencl_dispatch_udondnn_large_op_tensor(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, f32, f32, f32) -> i32
func.func private @__vrc_opencl_dispatch_rotary(i32, i32, i32, i32, i32) -> i32 attributes {
  vrc.synthesized_dispatch = @synthesized::@synthesized_rotary
}
func.func @rotate() {
  %bytes = arith.constant 524288 : i32
  %id1 = arith.constant 1 : i32
  %id2 = arith.constant 2 : i32
  %id3 = arith.constant 3 : i32
  %id4 = arith.constant 4 : i32
  %id5 = arith.constant 5 : i32
  %id6 = arith.constant 6 : i32
  %id7 = arith.constant 7 : i32
  %a = func.call @__vrc_opencl_buffer_alloc(%id1, %bytes) : (i32, i32) -> i32
  %b = func.call @__vrc_opencl_buffer_alloc(%id2, %bytes) : (i32, i32) -> i32
  %c = func.call @__vrc_opencl_buffer_alloc(%id3, %bytes) : (i32, i32) -> i32
  %x = func.call @__vrc_opencl_buffer_alloc(%id4, %bytes) : (i32, i32) -> i32
  %cos = func.call @__vrc_opencl_buffer_alloc(%id5, %bytes) : (i32, i32) -> i32
  %sin = func.call @__vrc_opencl_buffer_alloc(%id6, %bytes) : (i32, i32) -> i32
  %out = func.call @__vrc_opencl_buffer_alloc(%id7, %bytes) : (i32, i32) -> i32
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %count = arith.constant 256 : i32
  %alpha = arith.constant 1.0 : f32
  %beta = arith.constant 0.0 : f32
  %scaled = func.call @__vrc_opencl_dispatch_udondnn_large_op_tensor(%a, %b, %c, %x, %zero, %count, %count, %count, %zero, %zero, %zero, %one, %alpha, %alpha, %beta) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, f32, f32, f32) -> i32
  %rotated = func.call @__vrc_opencl_dispatch_rotary(%x, %cos, %sin, %out, %count) : (i32, i32, i32, i32, i32) -> i32
  return
}
