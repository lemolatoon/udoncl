// A Host copying one compiler-assigned Device buffer into another. The
// program and export are named in the module, as UDONCL_PROGRAM and
// UDONCL_EXPORT name them in Host C; the buffers are the textures the
// vrc.assigned_buffer operations name, so no config is needed.
module attributes {udoncl.program = "buffer_copy"} {
  func.func private @copy(!vrc.buffer, !vrc.buffer, i32, i32, i32) -> !vrc.event attributes {
    vrc.intrinsic = "buffer_copy"
  }
  func.func @copy_full() attributes {udoncl.export} {
    %bytes = arith.constant 4 : i32
    %zero = arith.constant 0 : i32
    %source = "vrc.assigned_buffer"() <{capacity_bytes = 4 : i64, format = "rgba8_unorm", resource_id = 0 : i32, texture_id = 1 : i32}> : () -> !vrc.buffer
    %destination = "vrc.assigned_buffer"() <{capacity_bytes = 4 : i64, format = "rgba8_unorm", resource_id = 1 : i32, texture_id = 2 : i32}> : () -> !vrc.buffer
    %event = func.call @copy(%source, %destination, %zero, %zero, %bytes) : (!vrc.buffer, !vrc.buffer, i32, i32, i32) -> !vrc.event
    return
  }
}
