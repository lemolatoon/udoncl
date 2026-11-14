module attributes {
  vrc.static_device_resource_discovery = {contexts = [{context_id = 9 : i32}, {context_id = 2 : i32}], schema_version = 1 : i32},
  vrc.static_device_resource_provenance = {discovery_schema_version = 1 : i32, schema_version = 1 : i32}
} {
  func.func private @clEnqueueWriteBuffer(i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
  func.func private @clEnqueueReadBuffer(i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
  func.func private @clEnqueueCopyBuffer(i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32

  func.func @run(%q: i32, %a: i32, %b: i32, %dynamic: i32) {
    %c0 = arith.constant 0 : i32
    %write = func.call @clEnqueueWriteBuffer(%q, %a, %c0, %c0, %c0, %c0, %c0, %c0, %c0) {vrc.static_device_operand_provenance = [{cases = [{call_path = [{callee = @helper, caller = @root_a, occurrence = 0 : i32}], may_be_null = false, resource_ids = [9 : i32], root = @root_a, unknown = false}, {call_path = [{callee = @helper, caller = @root_b, occurrence = 0 : i32}], may_be_null = false, resource_ids = [9 : i32], root = @root_b, unknown = false}], index = 1 : i32}]} : (i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    %read = func.call @clEnqueueReadBuffer(%q, %b, %c0, %c0, %c0, %c0, %c0, %c0, %c0) {vrc.static_device_operand_provenance = [{cases = [{call_path = [], may_be_null = false, resource_ids = [2 : i32], root = @run, unknown = false}], index = 1 : i32}]} : (i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    %copy = func.call @clEnqueueCopyBuffer(%q, %a, %b, %c0, %c0, %c0, %c0, %c0, %c0) {vrc.static_device_operand_provenance = [{cases = [{call_path = [], may_be_null = false, resource_ids = [9 : i32], root = @run, unknown = false}], index = 1 : i32}, {cases = [{call_path = [], may_be_null = false, resource_ids = [2 : i32], root = @run, unknown = false}], index = 2 : i32}]} : (i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    %specialized = func.call @clEnqueueWriteBuffer(%q, %a, %c0, %c0, %c0, %c0, %c0, %c0, %c0) {vrc.static_device_operand_provenance = [{cases = [{call_path = [{callee = @shared, caller = @root_a, occurrence = 0 : i32}], may_be_null = false, resource_ids = [9 : i32], root = @root_a, unknown = false}, {call_path = [{callee = @shared, caller = @root_b, occurrence = 1 : i32}], may_be_null = false, resource_ids = [2 : i32], root = @root_b, unknown = false}], index = 1 : i32}]} : (i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    %nullable = func.call @clEnqueueReadBuffer(%q, %dynamic, %c0, %c0, %c0, %c0, %c0, %c0, %c0) {vrc.static_device_operand_provenance = [{cases = [{call_path = [], may_be_null = true, resource_ids = [9 : i32], root = @run, unknown = false}], index = 1 : i32}]} : (i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    %not_allocation = func.call @clEnqueueWriteBuffer(%q, %dynamic, %c0, %c0, %c0, %c0, %c0, %c0, %c0) {vrc.static_device_operand_provenance = [{cases = [{call_path = [], may_be_null = false, resource_ids = [77 : i32], root = @run, unknown = false}], index = 1 : i32}]} : (i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    return
  }
}
