module {
  func.func @dispatch_square(%value: f32) {
    %id = arith.constant 1 : i32
    %bytes = arith.constant 4 : i32
    %output, %status = vrc.buffer_alloc %id, %bytes {format = "rgba8_unorm", usage = "storage"} : (i32, i32) -> (!vrc.buffer, i32)
    %desc = vrc.dispatch_desc_create %value {kernel_id = 1 : i32, scalar_kinds = ["f32"]} : (f32) -> !vrc.dispatch_desc
    %event = vrc.dispatch %desc(%output) {argument_kinds = ["buffer", "f32"], resource_accesses = ["write"], global_offset = array<i32: 0, 0, 0>, global_size = array<i32: 1, 1, 1>, kernel_id = 1 : i32, local_size = array<i32: 1, 1, 1>} : (!vrc.dispatch_desc, !vrc.buffer) -> !vrc.event
    return
  }
}
