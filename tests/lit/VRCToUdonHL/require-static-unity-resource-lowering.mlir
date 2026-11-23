// RUN: %vrc-opt --split-input-file --verify-diagnostics --vrc-require-static-unity-resource-lowering %s | %FileCheck %s

// An applied static Unity resource lowering, and a module without Device
// buffers, pass unchanged.

// CHECK-LABEL: module attributes {vrc.static_unity_resource_lowering_readiness = {applied = true
module attributes {vrc.static_unity_resource_lowering_readiness = {applied = true, pool_replacement_ready = true, schema_version = 1 : i32}} {
}

// -----

// CHECK-LABEL: func.func @scalar
module {
  func.func @scalar() {
    return
  }
}

// -----

// Stage 1 exception: a generated ONNX binding that banks sequence state
// keeps the run-time buffer pool.

// CHECK-LABEL: module attributes {udoncl.onnx_binding
module attributes {udoncl.onnx_binding = {physical = {enqueue_advance_sequence_state_symbol = "udon_onnx_binding_enqueue_advance_sequence_state"}}, vrc.static_unity_resource_lowering_readiness = {applied = false, dynamic_allocation_count = 2 : i32, pool_replacement_ready = false, schema_version = 1 : i32}} {
}

// -----

// expected-error@below {{every Unity Device buffer must resolve at compile time to one assigned buffer, but the static Unity resource lowering is not applied: 2 dynamic Device allocation(s); 1 unlowered OpenCL transfer call(s)}}
module attributes {udoncl.onnx_binding = {physical = {enqueue_advance_sequence_state_symbol = ""}}, vrc.static_device_resource_inventory = {resources = [{resource_id = 0 : i32}], schema_version = 1 : i32}, vrc.static_unity_resource_lowering_readiness = {applied = false, dynamic_allocation_count = 2 : i32, pool_replacement_ready = false, schema_version = 1 : i32, transfer_call_count = 1 : i32}} {
}

// -----

// expected-error@below {{every Unity Device buffer must resolve at compile time to one assigned buffer, but the static Unity resource lowering was not assessed}}
module {
  func.func @allocate() {
    %id = arith.constant 1 : i32
    %bytes = arith.constant 4 : i32
    %buffer, %status = vrc.buffer_alloc %id, %bytes {format = "rgba8_unorm", usage = "storage"} : (i32, i32) -> (!vrc.buffer, i32)
    return
  }
}

// -----

// A Host without Device buffers (here one that only reads a static tensor)
// passes even when the preview of the lowering did not apply.

// CHECK-LABEL: module attributes {vrc.static_device_resource_inventory
module attributes {vrc.static_device_resource_inventory = {resources = [], schema_version = 1 : i32}, vrc.static_unity_resource_lowering_readiness = {applied = false, pool_replacement_ready = false, schema_version = 1 : i32, unresolved_typed_transfer_operand_count = 1 : i32}} {
}
