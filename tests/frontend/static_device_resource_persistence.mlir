module attributes {
  vrc.static_device_resource_discovery = {
    contexts = [
      {context_id = 2 : i32, root = @Repeat},
      {context_id = 0 : i32, root = @RunLocal},
      {context_id = 1 : i32, root = @Initialize}
    ],
    schema_version = 1 : i32
  },
  vrc.static_device_resource_provenance = {
    discovery_schema_version = 1 : i32,
    schema_version = 1 : i32
  }
} {
  "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, initial_value = 0 : i32, linkage = "internal", storage_type = !udonhl.ptr<0>, sym_name = "saved"}> : () -> ()
  func.func private @consume(!udonhl.ptr<0>)

  func.func @RunLocal(%resource: !udonhl.ptr<0>) attributes {udoncl.export} {
    func.call @consume(%resource) {vrc.static_device_operand_provenance = [{cases = [{call_path = [], may_be_null = false, resource_ids = [0 : i32], root = @RunLocal, unknown = false}], index = 0 : i32}]} : (!udonhl.ptr<0>) -> ()
    return
  }

  func.func @Initialize(%resource: !udonhl.ptr<0>) attributes {udoncl.export} {
    func.call @consume(%resource) {vrc.static_device_operand_provenance = [{cases = [{call_path = [], may_be_null = false, resource_ids = [1 : i32], root = @Initialize, unknown = false}], index = 0 : i32}]} : (!udonhl.ptr<0>) -> ()
    return
  }

  func.func @Run(%resource: !udonhl.ptr<0>) attributes {udoncl.export} {
    func.call @consume(%resource) {vrc.static_device_operand_provenance = [{cases = [{call_path = [], may_be_null = false, resource_ids = [1 : i32], root = @Run, unknown = false}], index = 0 : i32}]} : (!udonhl.ptr<0>) -> ()
    return
  }

  func.func @Repeat(%resource: !udonhl.ptr<0>) attributes {udoncl.export} {
    %saved = udonhl.address_of @saved : !udonhl.ptr<0>
    "udonhl.store"(%resource, %saved) <{alignment = 4 : i64}> {vrc.static_device_operand_provenance = [{cases = [{call_path = [], may_be_null = false, resource_ids = [2 : i32], root = @Repeat, unknown = false}], index = 0 : i32}]} : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    return
  }
}
