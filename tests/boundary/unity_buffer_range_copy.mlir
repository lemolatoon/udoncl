// A Host copying one word into the middle of a larger compiler-assigned
// Device buffer; see unity_buffer_copy.mlir.
module attributes {udoncl.program = "buffer_range_copy"} {
  func.func private @copy(!vrc.buffer, !vrc.buffer, i32, i32, i32) -> !vrc.event attributes {
    vrc.intrinsic = "buffer_copy"
  }
  func.func @copy_word() attributes {udoncl.export} {
    %word = arith.constant 4 : i32
    %zero = arith.constant 0 : i32
    %offset = arith.constant 8 : i32
    %source = "vrc.assigned_buffer"() <{capacity_bytes = 4 : i64, format = "rgba8_unorm", resource_id = 0 : i32, texture_id = 1 : i32}> : () -> !vrc.buffer
    %destination = "vrc.assigned_buffer"() <{capacity_bytes = 16 : i64, format = "rgba8_unorm", resource_id = 1 : i32, texture_id = 2 : i32}> : () -> !vrc.buffer
    %event = func.call @copy(%source, %destination, %zero, %offset, %word) : (!vrc.buffer, !vrc.buffer, i32, i32, i32) -> !vrc.event
    return
  }
}
