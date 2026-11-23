// RUN: %vrc-opt --vrc-materialize-static-transfer-resources --split-input-file %s | %FileCheck %s

// Each transfer call whose snapshot names one allocation in every call
// context calls a specialization of the runtime transfer per allocation, in
// which the typed upload writes the assigned Texture. A call whose contexts
// name different allocations keeps the runtime transfer.

// CHECK: vrc.static_transfer_resource_materialization = {candidate_call_count = 4 : i32, materialized_call_count = 3 : i32, materialized_operand_count = 2 : i32, schema_version = 1 : i32, specialized_function_count = 2 : i32, unresolved_call_count = 1 : i32}
// CHECK-LABEL: func.func @clEnqueueWriteBuffer(
// CHECK: %[[TOKEN:.*]] = call @udoncl_mem_physical_token
// CHECK: %[[PHYSICAL:.*]] = builtin.unrealized_conversion_cast %[[TOKEN]]
// CHECK: vrc.buffer_upload %[[PHYSICAL]]
// CHECK-LABEL: func.func private @__vrc_static_clEnqueueWriteBuffer_1_0(
// CHECK-SAME: attributes {vrc.static_transfer_specialization = {callee = "clEnqueueWriteBuffer", resources = [{operand_index = 1 : i32, resource_id = 0 : i32}]}
// CHECK: %[[A:.*]] = "vrc.assigned_buffer"() <{capacity_bytes = 128 : i64, format = "rgba8_unorm", resource_id = 0 : i32, texture_id = 1 : i32}>
// CHECK: vrc.buffer_upload %[[A]]
// CHECK-LABEL: func.func private @__vrc_static_clEnqueueWriteBuffer_1_1(
// CHECK: %[[B:.*]] = "vrc.assigned_buffer"() <{capacity_bytes = 64 : i64, format = "rgba8_unorm", resource_id = 1 : i32, texture_id = 2 : i32}>
// CHECK: vrc.buffer_upload %[[B]]
// CHECK-LABEL: func.func @run
// CHECK: call @__vrc_static_clEnqueueWriteBuffer_1_0(
// CHECK: call @__vrc_static_clEnqueueWriteBuffer_1_1(
// CHECK: call @__vrc_static_clEnqueueWriteBuffer_1_0(
// CHECK: call @clEnqueueWriteBuffer(
module attributes {
  vrc.static_device_resource_inventory = {schema_version = 1 : i32, resources = [
    {capacity_bytes = 128 : i64, format = "rgba8_unorm", resource_id = 0 : i32, resource_kind = "allocation"},
    {capacity_bytes = 64 : i64, format = "rgba8_unorm", resource_id = 1 : i32, resource_kind = "allocation"}
  ]},
  vrc.unity_texture_assignments = {schema_version = 1 : i32, assignments = [
    {capacity_bytes = 128 : i64, format = "rgba8_unorm", resource_id = 0 : i32, resource_kind = "allocation", texture_id = 1 : i32},
    {capacity_bytes = 64 : i64, format = "rgba8_unorm", resource_id = 1 : i32, resource_kind = "allocation", texture_id = 2 : i32}
  ]}
} {
  func.func private @udoncl_mem_physical_token(!udonhl.ptr<0>) -> i32
  func.func @clEnqueueWriteBuffer(%queue: !udonhl.ptr<0>, %buffer: !udonhl.ptr<0>, %offset: i32, %bytes: i32, %source: i32) -> i32 {
    %token = func.call @udoncl_mem_physical_token(%buffer) : (!udonhl.ptr<0>) -> i32
    %physical = builtin.unrealized_conversion_cast %token : i32 to !vrc.buffer
    %event = vrc.buffer_upload %physical, %offset, %bytes, %source : (!vrc.buffer, i32, i32, i32) -> !vrc.event
    %status = builtin.unrealized_conversion_cast %event : !vrc.event to i32
    return %status : i32
  }
  func.func @run(%queue: !udonhl.ptr<0>, %a: !udonhl.ptr<0>, %b: !udonhl.ptr<0>, %either: !udonhl.ptr<0>, %source: i32) attributes {udoncl.export} {
    %zero = arith.constant 0 : i32
    %bytes = arith.constant 64 : i32
    %0 = func.call @clEnqueueWriteBuffer(%queue, %a, %zero, %bytes, %source) {vrc.static_transfer_binding_snapshot = {cases = [{call_path = [], resources = [{access = "write", operand_index = 1 : i32, resource_id = 0 : i32, resource_kind = "allocation"}], root = @run}], requires_context_specialization = false, schema_version = 1 : i32, transfer_kind = "write"}} : (!udonhl.ptr<0>, !udonhl.ptr<0>, i32, i32, i32) -> i32
    %1 = func.call @clEnqueueWriteBuffer(%queue, %b, %zero, %bytes, %source) {vrc.static_transfer_binding_snapshot = {cases = [{call_path = [], resources = [{access = "write", operand_index = 1 : i32, resource_id = 1 : i32, resource_kind = "allocation"}], root = @run}], requires_context_specialization = false, schema_version = 1 : i32, transfer_kind = "write"}} : (!udonhl.ptr<0>, !udonhl.ptr<0>, i32, i32, i32) -> i32
    %2 = func.call @clEnqueueWriteBuffer(%queue, %a, %zero, %bytes, %source) {vrc.static_transfer_binding_snapshot = {cases = [{call_path = [], resources = [{access = "write", operand_index = 1 : i32, resource_id = 0 : i32, resource_kind = "allocation"}], root = @run}], requires_context_specialization = false, schema_version = 1 : i32, transfer_kind = "write"}} : (!udonhl.ptr<0>, !udonhl.ptr<0>, i32, i32, i32) -> i32
    %3 = func.call @clEnqueueWriteBuffer(%queue, %either, %zero, %bytes, %source) {vrc.static_transfer_binding_snapshot = {cases = [{call_path = [], resources = [{access = "write", operand_index = 1 : i32, resource_id = 0 : i32, resource_kind = "allocation"}], root = @run}, {call_path = [], resources = [{access = "write", operand_index = 1 : i32, resource_id = 1 : i32, resource_kind = "allocation"}], root = @other}], requires_context_specialization = true, schema_version = 1 : i32, transfer_kind = "write"}} : (!udonhl.ptr<0>, !udonhl.ptr<0>, i32, i32, i32) -> i32
    return
  }
}

// -----

// Once every call reaches a specialization the runtime transfer is erased;
// a readback reads the assigned Texture of its source.

// CHECK-NOT: func.func @clEnqueueReadBuffer(
// CHECK-LABEL: func.func private @__vrc_static_clEnqueueReadBuffer_1_0(
// CHECK: %[[A:.*]] = "vrc.assigned_buffer"() <{capacity_bytes = 128 : i64, format = "rgba8_unorm", resource_id = 0 : i32, texture_id = 1 : i32}>
// CHECK: vrc.buffer_readback %[[A]]
// CHECK-LABEL: func.func @read
// CHECK: call @__vrc_static_clEnqueueReadBuffer_1_0(
module attributes {
  vrc.static_device_resource_inventory = {schema_version = 1 : i32, resources = [
    {capacity_bytes = 128 : i64, format = "rgba8_unorm", resource_id = 0 : i32, resource_kind = "allocation"}
  ]},
  vrc.unity_texture_assignments = {schema_version = 1 : i32, assignments = [
    {capacity_bytes = 128 : i64, format = "rgba8_unorm", resource_id = 0 : i32, resource_kind = "allocation", texture_id = 1 : i32}
  ]}
} {
  func.func private @udoncl_mem_physical_token(!udonhl.ptr<0>) -> i32
  func.func @clEnqueueReadBuffer(%queue: !udonhl.ptr<0>, %buffer: !udonhl.ptr<0>, %offset: i32, %bytes: i32, %destination: i32) -> i32 {
    %token = func.call @udoncl_mem_physical_token(%buffer) : (!udonhl.ptr<0>) -> i32
    %physical = builtin.unrealized_conversion_cast %token : i32 to !vrc.buffer
    %event = vrc.buffer_readback %physical, %offset, %bytes, %destination : (!vrc.buffer, i32, i32, i32) -> !vrc.event
    %status = builtin.unrealized_conversion_cast %event : !vrc.event to i32
    return %status : i32
  }
  func.func @read(%queue: !udonhl.ptr<0>, %a: !udonhl.ptr<0>, %destination: i32) attributes {udoncl.export} {
    %zero = arith.constant 0 : i32
    %bytes = arith.constant 64 : i32
    %0 = func.call @clEnqueueReadBuffer(%queue, %a, %zero, %bytes, %destination) {vrc.static_transfer_binding_snapshot = {cases = [{call_path = [], resources = [{access = "read", operand_index = 1 : i32, resource_id = 0 : i32, resource_kind = "allocation"}], root = @read}], requires_context_specialization = false, schema_version = 1 : i32, transfer_kind = "read"}} : (!udonhl.ptr<0>, !udonhl.ptr<0>, i32, i32, i32) -> i32
    return
  }
}
