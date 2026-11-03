module {
  func.func private @allocate(i32, i32) -> (!vrc.buffer, i32) attributes {
    vrc.format = "rgba8_unorm",
    vrc.intrinsic = "buffer_alloc",
    vrc.usage = "storage"
  }
  func.func private @copy(!vrc.buffer, !vrc.buffer, i32, i32, i32) -> !vrc.event attributes {
    vrc.intrinsic = "buffer_copy"
  }
  func.func @copy_full() {
    %bytes = arith.constant 4 : i32
    %zero = arith.constant 0 : i32
    %source_id = arith.constant 1 : i32
    %destination_id = arith.constant 2 : i32
    %source, %source_status = func.call @allocate(%source_id, %bytes) : (i32, i32) -> (!vrc.buffer, i32)
    %destination, %destination_status = func.call @allocate(%destination_id, %bytes) : (i32, i32) -> (!vrc.buffer, i32)
    %event = func.call @copy(%source, %destination, %zero, %zero, %bytes) : (!vrc.buffer, !vrc.buffer, i32, i32, i32) -> !vrc.event
    return
  }
}
