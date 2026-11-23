// Host half of the generated kernel channel test. unity_synthesized_dispatch.py
// appends it to synthesized_rotary.mlir, whose gpu.module @synthesized carries
// the kernel that @__vrc_opencl_dispatch_rotary names. The Host scales a
// tensor with the UdonDNN OpTensor library kernel and then rotates the result
// with the synthesized kernel. Its Device buffers are compiler-assigned
// textures (vrc.assigned_buffer) whose tokens the dispatches take, and the
// test wraps the module in the program attribute UDONCL_PROGRAM sets, so no
// config is needed.
func.func private @__vrc_opencl_dispatch_udondnn_large_op_tensor(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, f32, f32, f32) -> i32
func.func private @__vrc_opencl_dispatch_rotary(i32, i32, i32, i32, i32) -> i32 attributes {
  vrc.synthesized_dispatch = @synthesized::@synthesized_rotary
}
func.func @rotate() attributes {udoncl.export} {
  %a_buffer = "vrc.assigned_buffer"() <{capacity_bytes = 524288 : i64, format = "rgba8_unorm", resource_id = 0 : i32, texture_id = 1 : i32}> : () -> !vrc.buffer
  %a = builtin.unrealized_conversion_cast %a_buffer : !vrc.buffer to i32
  %b_buffer = "vrc.assigned_buffer"() <{capacity_bytes = 524288 : i64, format = "rgba8_unorm", resource_id = 1 : i32, texture_id = 2 : i32}> : () -> !vrc.buffer
  %b = builtin.unrealized_conversion_cast %b_buffer : !vrc.buffer to i32
  %c_buffer = "vrc.assigned_buffer"() <{capacity_bytes = 524288 : i64, format = "rgba8_unorm", resource_id = 2 : i32, texture_id = 3 : i32}> : () -> !vrc.buffer
  %c = builtin.unrealized_conversion_cast %c_buffer : !vrc.buffer to i32
  %x_buffer = "vrc.assigned_buffer"() <{capacity_bytes = 524288 : i64, format = "rgba8_unorm", resource_id = 3 : i32, texture_id = 4 : i32}> : () -> !vrc.buffer
  %x = builtin.unrealized_conversion_cast %x_buffer : !vrc.buffer to i32
  %cos_buffer = "vrc.assigned_buffer"() <{capacity_bytes = 524288 : i64, format = "rgba8_unorm", resource_id = 4 : i32, texture_id = 5 : i32}> : () -> !vrc.buffer
  %cos = builtin.unrealized_conversion_cast %cos_buffer : !vrc.buffer to i32
  %sin_buffer = "vrc.assigned_buffer"() <{capacity_bytes = 524288 : i64, format = "rgba8_unorm", resource_id = 5 : i32, texture_id = 6 : i32}> : () -> !vrc.buffer
  %sin = builtin.unrealized_conversion_cast %sin_buffer : !vrc.buffer to i32
  %out_buffer = "vrc.assigned_buffer"() <{capacity_bytes = 524288 : i64, format = "rgba8_unorm", resource_id = 6 : i32, texture_id = 7 : i32}> : () -> !vrc.buffer
  %out = builtin.unrealized_conversion_cast %out_buffer : !vrc.buffer to i32
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %count = arith.constant 256 : i32
  %alpha = arith.constant 1.0 : f32
  %beta = arith.constant 0.0 : f32
  %scaled = func.call @__vrc_opencl_dispatch_udondnn_large_op_tensor(%a, %b, %c, %x, %zero, %count, %count, %count, %zero, %zero, %zero, %one, %alpha, %alpha, %beta) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, f32, f32, f32) -> i32
  %rotated = func.call @__vrc_opencl_dispatch_rotary(%x, %cos, %sin, %out, %count) : (i32, i32, i32, i32, i32) -> i32
  return
}
