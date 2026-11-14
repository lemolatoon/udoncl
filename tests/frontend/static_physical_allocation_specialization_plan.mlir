module attributes {
  vrc.static_device_resource_inventory = {resources = [
    {capacity_bytes = 16 : i64, format = "rgba8_unorm", persistent = true, resource_id = 0 : i32, resource_kind = "allocation"},
    {capacity_bytes = 32 : i64, format = "rgba8_unorm", persistent = true, resource_id = 1 : i32, resource_kind = "allocation"}
  ], schema_version = 1 : i32},
  vrc.unity_texture_assignments = {assignments = [
    {capacity_bytes = 16 : i64, format = "rgba8_unorm", persistent = true, resource_id = 0 : i32, resource_kind = "allocation", texture_id = 1 : i32},
    {capacity_bytes = 32 : i64, format = "rgba8_unorm", persistent = true, resource_id = 1 : i32, resource_kind = "allocation", texture_id = 2 : i32}
  ], schema_version = 1 : i32}
} {
  func.func @root() -> i32 {
    %first = call @create() {vrc.static_device_allocation_site = {function = @root, occurrence = 0 : i32, primitive = "create"}, vrc.static_device_result_provenance = [{cases = [{resource_ids = [0 : i32], unknown = false}], index = 0 : i32}]} : () -> i32
    %second = call @create() {vrc.static_device_allocation_site = {function = @root, occurrence = 1 : i32, primitive = "create"}, vrc.static_device_result_provenance = [{cases = [{resource_ids = [1 : i32], unknown = false}], index = 0 : i32}]} : () -> i32
    return %second : i32
  }

  func.func private @create() -> i32 {
    %result = call @physical() : () -> i32
    return %result : i32
  }

  func.func private @physical() -> i32 {
    %logical = arith.constant 1 : i32
    %bytes = arith.constant 32 : i32
    %buffer, %status = vrc.buffer_alloc %logical, %bytes {format = "rgba8_unorm", usage = "storage"} : (i32, i32) -> (!vrc.buffer, i32)
    return %status : i32
  }
}

// -----

module attributes {
  vrc.static_device_resource_inventory = {resources = [{capacity_bytes = 16 : i64, format = "rgba8_unorm", persistent = false, resource_id = 0 : i32, resource_kind = "allocation"}], schema_version = 1 : i32},
  vrc.unity_texture_assignments = {assignments = [{capacity_bytes = 16 : i64, format = "rgba8_unorm", persistent = false, resource_id = 0 : i32, resource_kind = "allocation", texture_id = 1 : i32}], schema_version = 1 : i32}
} {
  func.func @nonpersistent() -> i32 {
    %result = call @create() {vrc.static_device_allocation_site = {function = @nonpersistent, occurrence = 0 : i32, primitive = "create"}, vrc.static_device_result_provenance = [{cases = [{resource_ids = [0 : i32], unknown = false}], index = 0 : i32}]} : () -> i32
    return %result : i32
  }

  func.func private @create() -> i32 {
    %logical = arith.constant 1 : i32
    %bytes = arith.constant 16 : i32
    %buffer, %status = vrc.buffer_alloc %logical, %bytes {format = "rgba8_unorm", usage = "storage"} : (i32, i32) -> (!vrc.buffer, i32)
    return %status : i32
  }
}

// -----

module attributes {
  vrc.static_device_resource_inventory = {resources = [{capacity_bytes = 16 : i64, format = "rgba8_unorm", persistent = true, resource_id = 0 : i32, resource_kind = "allocation"}], schema_version = 1 : i32},
  vrc.unity_texture_assignments = {assignments = [{capacity_bytes = 16 : i64, format = "rgba8_unorm", persistent = true, resource_id = 0 : i32, resource_kind = "allocation", texture_id = 1 : i32}], schema_version = 1 : i32}
} {
  func.func @blocked() -> i32 {
    %result = call @create() {vrc.static_device_allocation_site = {function = @blocked, occurrence = 0 : i32, primitive = "create"}} : () -> i32
    return %result : i32
  }

  func.func private @create() -> i32 {
    %logical = arith.constant 1 : i32
    %bytes = arith.constant 16 : i32
    %buffer, %status = vrc.buffer_alloc %logical, %bytes {format = "rgba8_unorm", usage = "storage"} : (i32, i32) -> (!vrc.buffer, i32)
    return %status : i32
  }
}
