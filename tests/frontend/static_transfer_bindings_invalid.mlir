module attributes {vrc.static_device_resource_discovery = {contexts = [{context_id = 1 : i32}], schema_version = 1 : i32}, vrc.static_device_resource_provenance = {discovery_schema_version = 1 : i32, schema_version = 1 : i32}} {
  func.func private @clEnqueueWriteBuffer(i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
  func.func @duplicate(%q: i32, %buffer: i32) {
    %c0 = arith.constant 0 : i32
    // expected-error @below {{OpenCL transfer provenance is malformed}}
    %status = func.call @clEnqueueWriteBuffer(%q, %buffer, %c0, %c0, %c0, %c0, %c0, %c0, %c0) {vrc.static_device_operand_provenance = [{cases = [], index = 1 : i32}, {cases = [], index = 1 : i32}]} : (i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    return
  }
}

// -----

module attributes {vrc.static_device_resource_discovery = {contexts = [{context_id = 1 : i32}], schema_version = 1 : i32}, vrc.static_device_resource_provenance = {discovery_schema_version = 1 : i32, schema_version = 1 : i32}} {
  func.func private @clEnqueueReadBuffer(i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
  func.func @duplicate_context(%q: i32, %buffer: i32) {
    %c0 = arith.constant 0 : i32
    // expected-error @below {{OpenCL transfer provenance has duplicate call contexts}}
    %status = func.call @clEnqueueReadBuffer(%q, %buffer, %c0, %c0, %c0, %c0, %c0, %c0, %c0) {vrc.static_device_operand_provenance = [{cases = [{call_path = [], may_be_null = false, resource_ids = [1 : i32], root = @duplicate_context, unknown = false}, {call_path = [], may_be_null = false, resource_ids = [1 : i32], root = @duplicate_context, unknown = false}], index = 1 : i32}]} : (i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    return
  }
}

// -----

module attributes {vrc.static_device_resource_discovery = {contexts = [{context_id = 1 : i32}], schema_version = 1 : i32}, vrc.static_device_resource_provenance = {discovery_schema_version = 1 : i32, schema_version = 1 : i32}} {
  func.func private @clEnqueueCopyBuffer(i32, i32) -> i32
  func.func @bad_signature(%a: i32, %b: i32) {
    // expected-error @below {{OpenCL transfer call has an invalid signature}}
    %status = func.call @clEnqueueCopyBuffer(%a, %b) : (i32, i32) -> i32
    return
  }
}
