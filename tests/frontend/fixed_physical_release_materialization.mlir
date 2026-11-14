module attributes {vrc.target = "vrc-pc"} {
  func.func @finalize(%token: i32) -> i32 {
    %buffer = builtin.unrealized_conversion_cast %token : i32 to !vrc.buffer
    %status = vrc.buffer_release %buffer : (!vrc.buffer) -> i32
    return %status : i32
  }
}
