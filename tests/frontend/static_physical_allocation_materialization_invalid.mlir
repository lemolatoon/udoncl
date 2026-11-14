module attributes {
  vrc.static_device_resource_inventory = {resources = [{capacity_bytes = 16 : i64, format = "rgba8_unorm", persistent = true, resource_id = 0 : i32, resource_kind = "allocation"}], schema_version = 1 : i32},
  vrc.static_physical_allocation_specialization = {annotated_sink_count = 1 : i32, blocker_count = 0 : i32, cloned_function_count = 1 : i32, rewritten_call_count = 1 : i32, schema_version = 1 : i32, specialized_context_count = 1 : i32},
  vrc.unity_texture_assignments = {assignments = [{capacity_bytes = 16 : i64, format = "rgba8_unorm", persistent = true, resource_id = 0 : i32, resource_kind = "allocation", texture_id = 1 : i32}], schema_version = 1 : i32}
} {
  func.func private @physical() -> i32 {
    %logical = arith.constant 1 : i32
    %bytes = arith.constant 8 : i32
    // expected-error @below {{constant physical allocation size does not match assignment}}
    %buffer, %status = vrc.buffer_alloc %logical, %bytes {format = "rgba8_unorm", usage = "storage", vrc.static_physical_allocation_assignment = {capacity_bytes = 16 : i64, format = "rgba8_unorm", resource_id = 0 : i32, schema_version = 1 : i32, texture_id = 1 : i32}} : (i32, i32) -> (!vrc.buffer, i32)
    return %status : i32
  }
}
