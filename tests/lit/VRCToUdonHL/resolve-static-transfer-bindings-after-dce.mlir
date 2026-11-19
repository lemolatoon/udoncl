// RUN: %vrc-opt --vrc-resolve-static-transfer-bindings %s | %FileCheck %s --check-prefix=BEFORE
// RUN: %vrc-opt --vrc-resolve-static-transfer-bindings --symbol-dce --vrc-resolve-static-transfer-bindings %s | %FileCheck %s --check-prefix=AFTER

// The transfer binding summary counts the raw transfers present when the pass
// runs. Unreachable library helpers may still hold transfers at that point;
// after they are erased, re-running the pass recomputes a summary that matches
// the surviving IR, which later readiness assessment requires.

// BEFORE: vrc.static_transfer_bindings = {candidate_count = 2 : i32, requires_context_specialization_count = 0 : i32, resolved_count = 1 : i32, schema_version = 1 : i32, unresolved_count = 1 : i32}
// AFTER: vrc.static_transfer_bindings = {candidate_count = 1 : i32, requires_context_specialization_count = 0 : i32, resolved_count = 1 : i32, schema_version = 1 : i32, unresolved_count = 0 : i32}
// AFTER-NOT: @unreachable_helper
module attributes {
  vrc.static_device_resource_discovery = {contexts = [{context_id = 9 : i32}], schema_version = 1 : i32},
  vrc.static_device_resource_provenance = {discovery_schema_version = 1 : i32, schema_version = 1 : i32}
} {
  func.func private @clEnqueueWriteBuffer(i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32

  func.func @run(%q: i32, %a: i32) {
    %c0 = arith.constant 0 : i32
    %write = func.call @clEnqueueWriteBuffer(%q, %a, %c0, %c0, %c0, %c0, %c0, %c0, %c0) {vrc.static_device_operand_provenance = [{cases = [{call_path = [], may_be_null = false, resource_ids = [9 : i32], root = @run, unknown = false}], index = 1 : i32}]} : (i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    return
  }

  func.func private @unreachable_helper(%q: i32, %a: i32) {
    %c0 = arith.constant 0 : i32
    %write = func.call @clEnqueueWriteBuffer(%q, %a, %c0, %c0, %c0, %c0, %c0, %c0, %c0) : (i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    return
  }
}
