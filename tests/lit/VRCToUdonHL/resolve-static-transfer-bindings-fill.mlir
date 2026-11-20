// RUN: %vrc-opt --vrc-resolve-static-transfer-bindings %s | %FileCheck %s

// clEnqueueFillBuffer writes its buffer operand (index 1) like
// clEnqueueWriteBuffer, so a fill whose buffer has one allocation context is
// a resolved "fill" transfer and an unproven one stays unresolved.

// CHECK: vrc.static_transfer_bindings = {candidate_count = 2 : i32, requires_context_specialization_count = 0 : i32, resolved_count = 1 : i32, schema_version = 1 : i32, unresolved_count = 1 : i32}
// CHECK-LABEL: func.func @run(
// CHECK: call @clEnqueueFillBuffer
// CHECK-SAME: vrc.static_transfer_binding_snapshot = {cases = [{call_path = [], resources = [{access = "write", operand_index = 1 : i32, resource_id = 9 : i32, resource_kind = "allocation"}], root = @run}], requires_context_specialization = false, schema_version = 1 : i32, transfer_kind = "fill"}
// CHECK: call @clEnqueueFillBuffer
// CHECK-NOT: vrc.static_transfer_binding_snapshot
// CHECK: return
module attributes {
  vrc.static_device_resource_discovery = {contexts = [{context_id = 9 : i32}], schema_version = 1 : i32},
  vrc.static_device_resource_provenance = {discovery_schema_version = 1 : i32, schema_version = 1 : i32}
} {
  func.func private @clEnqueueFillBuffer(i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32

  func.func @run(%q: i32, %a: i32, %b: i32) {
    %c0 = arith.constant 0 : i32
    %c4 = arith.constant 4 : i32
    %resolved = func.call @clEnqueueFillBuffer(%q, %a, %c0, %c4, %c0, %c4, %c0, %c0, %c0) {vrc.static_device_operand_provenance = [{cases = [{call_path = [], may_be_null = false, resource_ids = [9 : i32], root = @run, unknown = false}], index = 1 : i32}]} : (i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    %unresolved = func.call @clEnqueueFillBuffer(%q, %b, %c0, %c4, %c0, %c4, %c0, %c0, %c0) : (i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    return
  }
}
