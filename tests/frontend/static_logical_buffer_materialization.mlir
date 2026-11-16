module attributes {
  vrc.static_unity_buffer_reconciliation = {schema_version = 1 : i32, buffers = [
    {buffer_id = 2 : i32, heap_slot = "__udoncl_buffer_2", resolution = "static_alias", resource_id = 7 : i32, resource_kind = "allocation", texture_id = 3 : i32},
    {buffer_id = 5 : i32, heap_slot = "__udoncl_buffer_5", resolution = "static_alias", resource_id = 9 : i32, resource_kind = "allocation", texture_id = 4 : i32}
  ]},
  vrc.static_unity_buffer_layout_plan = {schema_version = 1 : i32, ready = true, buffers = [
    {buffer_id = 3 : i32, capacity_bytes = 64 : i64, format = "rgba8_unorm", heap_slot = "__udoncl_buffer_3", persistent = true, public = true, resource_id = 7 : i32, resource_kind = "allocation", texture_id = 3 : i32},
    {buffer_id = 4 : i32, capacity_bytes = 128 : i64, format = "rgba8_unorm", heap_slot = "__udoncl_buffer_4", persistent = true, public = true, resource_id = 9 : i32, resource_kind = "allocation", texture_id = 4 : i32}
  ]}
} {
  func.func @copy(%dynamic_id: i32) -> (!vrc.event, i32) {
    %c2 = arith.constant 2 : i32
    %c5 = arith.constant 5 : i32
    %c32 = arith.constant 32 : i32
    %c64 = arith.constant 64 : i32
    %source, %source_status = vrc.buffer_alloc %c5, %c64 {format = "rgba8_unorm", usage = "storage"} : (i32, i32) -> (!vrc.buffer, i32)
    %destination, %destination_status = vrc.buffer_alloc %c2, %c32 {format = "rgba8_unorm", usage = "storage"} : (i32, i32) -> (!vrc.buffer, i32)
    %dynamic, %dynamic_status = vrc.buffer_alloc %dynamic_id, %c32 {format = "rgba8_unorm", usage = "storage"} : (i32, i32) -> (!vrc.buffer, i32)
    %copy = vrc.buffer_copy %source, %destination, %source_status, %destination_status, %c32 : (!vrc.buffer, !vrc.buffer, i32, i32, i32) -> !vrc.event
    vrc.buffer_release %dynamic : (!vrc.buffer) -> i32
    return %copy, %dynamic_status : !vrc.event, i32
  }
}
