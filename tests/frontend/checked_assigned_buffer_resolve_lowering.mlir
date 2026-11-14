module {
  func.func @run(%logical: !udonhl.ptr<0>) -> i32 {
    %four = arith.constant 4 : i32
    %output = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %assigned = "vrc.assigned_buffer"() {capacity_bytes = 16 : i64, format = "rgba8_unorm", resource_id = 3 : i32, texture_id = 2 : i32} : () -> !vrc.buffer
    %status = "udonhl.checked_assigned_buffer_resolve"(%logical, %assigned, %output) : (!udonhl.ptr<0>, !vrc.buffer, !udonhl.ptr<0>) -> i32
    return %status : i32
  }
}
