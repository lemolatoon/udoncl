module attributes {
  vrc.static_device_resource_inventory = {resources = [
    {capacity_bytes = 64 : i64, format = "rgba8_unorm", persistent = true, resource_id = 7 : i32, resource_kind = "allocation"},
    {capacity_bytes = 32 : i64, format = "rgba8_unorm", persistent = true, resource_id = 8 : i32, resource_kind = "allocation"}
  ], schema_version = 1 : i32},
  vrc.unity_texture_assignments = {assignments = [
    {capacity_bytes = 64 : i64, format = "rgba8_unorm", persistent = true, resource_id = 7 : i32, resource_kind = "allocation", texture_id = 2 : i32},
    {capacity_bytes = 32 : i64, format = "rgba8_unorm", persistent = true, resource_id = 8 : i32, resource_kind = "allocation", texture_id = 1 : i32}
  ], schema_version = 1 : i32}
} {
  func.func private @udonclResolvePhysicalBuffer(!udonhl.ptr<0>, !udonhl.ptr<0>) -> i32

  func.func @Run(%proven: !udonhl.ptr<0>, %nullable: !udonhl.ptr<0>) attributes {udoncl.export} {
    %four = arith.constant 4 : i32
    %proven_token = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %proven_status = func.call @udonclResolvePhysicalBuffer(%proven, %proven_token) {
      vrc.static_device_operand_provenance = [{
        cases = [{call_path = [], may_be_null = false, resource_ids = [7 : i32], root = @Run, unknown = false}],
        index = 0 : i32
      }]
    } : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    %proven_value = "udonhl.load"(%proven_token) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> i32

    %nullable_token = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %nullable_status = func.call @udonclResolvePhysicalBuffer(%nullable, %nullable_token) {
      vrc.static_device_operand_provenance = [{
        cases = [{call_path = [], may_be_null = true, resource_ids = [8 : i32], root = @Run, unknown = false}],
        index = 0 : i32
      }]
    } : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    return
  }
}
