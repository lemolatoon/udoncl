// RUN: udoncl-opt --vrc-analyze-static-device-resource-persistence --split-input-file --verify-diagnostics %s

module attributes {
  vrc.static_device_resource_discovery = {contexts = [{context_id = 0 : i32, root = @Run}], schema_version = 1 : i32},
  vrc.static_device_resource_provenance = {discovery_schema_version = 1 : i32, schema_version = 1 : i32}
} {
  func.func private @consume(i32)
  func.func @Run(%resource: i32) attributes {udoncl.export} {
    // expected-error@+1 {{static resource persistence provenance is unknown}}
    func.call @consume(%resource) {vrc.static_device_operand_provenance = [{cases = [{call_path = [], may_be_null = true, resource_ids = [], root = @Run, unknown = true}], index = 0 : i32}]} : (i32) -> ()
    return
  }
}

// -----

module attributes {
  vrc.static_device_resource_discovery = {contexts = [{context_id = 0 : i32, root = @Run}], schema_version = 1 : i32},
  vrc.static_device_resource_provenance = {discovery_schema_version = 1 : i32, schema_version = 1 : i32}
} {
  func.func private @consume(i32)
  func.func @Run(%resource: i32) attributes {udoncl.export} {
    // expected-error@+1 {{provenance references unknown allocation context}}
    func.call @consume(%resource) {vrc.static_device_operand_provenance = [{cases = [{call_path = [], may_be_null = false, resource_ids = [7 : i32], root = @Run, unknown = false}], index = 0 : i32}]} : (i32) -> ()
    return
  }
}
