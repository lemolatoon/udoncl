// RUN: %vrc-opt --vrc-resolve-static-transfer-bindings %s | %FileCheck %s

// A Host variable that holds a Device buffer may be null (before the Host
// creates the buffer, or after it releases it), so a transfer of the handle
// loaded from it has a nullable case. It still names exactly one allocation
// when it is not null, and the OpenCL transfer rejects a null handle before it
// moves any data, so the call resolves like a non-null one. A case that may
// only be null names no allocation and stays unresolved.

// CHECK: vrc.static_transfer_bindings = {candidate_count = 2 : i32, requires_context_specialization_count = 0 : i32, resolved_count = 1 : i32, schema_version = 1 : i32, unresolved_count = 1 : i32}
// CHECK-LABEL: func.func @commit(
// CHECK: call @clEnqueueCopyBuffer
// CHECK-SAME: vrc.static_transfer_binding_snapshot = {cases = [{call_path = [], resources = [{access = "read", operand_index = 1 : i32, resource_id = 5 : i32, resource_kind = "allocation"}, {access = "write", operand_index = 2 : i32, resource_id = 2 : i32, resource_kind = "allocation"}], root = @commit}], requires_context_specialization = false, schema_version = 1 : i32, transfer_kind = "copy"}
// CHECK: call @clEnqueueCopyBuffer
// CHECK-NOT: vrc.static_transfer_binding_snapshot
// CHECK: return
module attributes {
  vrc.static_device_resource_discovery = {contexts = [{context_id = 2 : i32}, {context_id = 5 : i32}], schema_version = 1 : i32},
  vrc.static_device_resource_provenance = {discovery_schema_version = 1 : i32, schema_version = 1 : i32}
} {
  func.func private @clEnqueueCopyBuffer(i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32

  func.func @commit(%q: i32, %present: i32, %past: i32, %null: i32) {
    %c0 = arith.constant 0 : i32
    %c128 = arith.constant 128 : i32
    %nullable = func.call @clEnqueueCopyBuffer(%q, %present, %past, %c0, %c0, %c128, %c0, %c0, %c0) {vrc.static_device_operand_provenance = [{cases = [{call_path = [], may_be_null = true, resource_ids = [5 : i32], root = @commit, unknown = false}], index = 1 : i32}, {cases = [{call_path = [], may_be_null = true, resource_ids = [2 : i32], root = @commit, unknown = false}], index = 2 : i32}]} : (i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    %only_null = func.call @clEnqueueCopyBuffer(%q, %null, %past, %c0, %c0, %c128, %c0, %c0, %c0) {vrc.static_device_operand_provenance = [{cases = [{call_path = [], may_be_null = true, resource_ids = [], root = @commit, unknown = false}], index = 1 : i32}, {cases = [{call_path = [], may_be_null = true, resource_ids = [2 : i32], root = @commit, unknown = false}], index = 2 : i32}]} : (i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    return
  }
}
