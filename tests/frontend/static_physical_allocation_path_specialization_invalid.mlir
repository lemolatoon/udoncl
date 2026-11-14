// expected-error @below {{planned physical allocation call is missing or not private}}
module attributes {
  vrc.static_physical_allocation_specialization_plan = {
    blocker_count = 0 : i32,
    contexts = [{call_path = [{callee = @public_create, caller = @root, occurrence = 0 : i32}], capacity_bytes = 16 : i64, format = "rgba8_unorm", resource_id = 0 : i32, texture_id = 1 : i32}],
    planned_count = 1 : i32,
    schema_version = 1 : i32
  }
} {
  func.func @root() -> i32 {
    %result = call @public_create() : () -> i32
    return %result : i32
  }

  func.func @public_create() -> i32 {
    %logical = arith.constant 1 : i32
    %bytes = arith.constant 16 : i32
    %buffer, %status = vrc.buffer_alloc %logical, %bytes {format = "rgba8_unorm", usage = "storage"} : (i32, i32) -> (!vrc.buffer, i32)
    return %status : i32
  }
}
