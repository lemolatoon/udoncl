// RUN: %vrc-opt --split-input-file --verify-diagnostics --vrc-report-static-unity-resource-lowering %s | %FileCheck %s

// The report leaves the module unchanged.
// CHECK: vrc.static_unity_resource_lowering_readiness

// expected-remark@+1 {{static Unity resource lowering applied}}
module attributes {vrc.static_unity_resource_lowering_readiness = {
    applied = true, dispatch_ready = true, dynamic_allocation_count = 0 : i32,
    dynamic_release_count = 0 : i32, dynamic_resolve_count = 0 : i32,
    pool_replacement_ready = true, schema_version = 1 : i32,
    transfer_call_count = 0 : i32,
    unresolved_dispatch_operand_count = 0 : i32,
    unresolved_typed_transfer_operand_count = 0 : i32}} {
}

// -----

// Every reason the canonical pool does not replace the configured one is
// reported: a public allocation without alias, a configured public buffer
// without resource, a public resource without configured buffer, and the
// Device commands that still name a resource at run time.

// expected-remark@+8 {{static Unity resource lowering not applied}}
// expected-remark@+7 {{static Unity resource lowering blocker: public resource 2 (logits) has no public alias}}
// expected-remark@+6 {{static Unity resource lowering blocker: configured public buffer 3 (input 0) resolves to no static public resource}}
// expected-remark@+5 {{static Unity resource lowering blocker: static public resource 1 (output 0) has no configured buffer}}
// expected-remark@+4 {{static Unity resource lowering blocker: the canonical buffer layout has 3 blocker(s)}}
// expected-remark@+3 {{static Unity resource lowering blocker: 1 dynamic Device allocation(s)}}
// expected-remark@+2 {{static Unity resource lowering blocker: 2 unlowered OpenCL transfer call(s)}}
// expected-remark@+1 {{static Unity resource lowering blocker: Device dispatches are not ready}}
module attributes {
    vrc.static_device_resource_inventory = {schema_version = 1 : i32, resources = [
      {resource_id = 0 : i32, resource_kind = "allocation", public = false},
      {resource_id = 1 : i32, resource_kind = "allocation", public = true, name = "image"},
      {resource_id = 2 : i32, resource_kind = "allocation", public = true, name = "logits"}]},
    vrc.static_public_resource_aliases = {schema_version = 1 : i32, aliases = [
      {resource_id = 1 : i32, resource_kind = "allocation", public_access = "output", public_index = 0 : i32}]},
    vrc.static_unity_buffer_reconciliation = {schema_version = 1 : i32, buffers = [
      {buffer_id = 3 : i32, heap_slot = "__udoncl_buffer_3", public_access = "input", public_index = 0 : i32, resolution = "legacy_public"},
      {buffer_id = 4 : i32, heap_slot = "__udoncl_buffer_4", resolution = "legacy_private"}]},
    vrc.static_unity_buffer_layout_plan = {schema_version = 1 : i32, ready = false, blocker_count = 3 : i32},
    vrc.static_unity_resource_lowering_readiness = {
      applied = false, dispatch_ready = false, dynamic_allocation_count = 1 : i32,
      dynamic_release_count = 0 : i32, dynamic_resolve_count = 0 : i32,
      pool_replacement_ready = false, schema_version = 1 : i32,
      transfer_call_count = 2 : i32,
      unresolved_dispatch_operand_count = 0 : i32,
      unresolved_typed_transfer_operand_count = 0 : i32}} {
}

// -----

// expected-remark@+1 {{static Unity resource lowering was not assessed}}
module {
}

// -----

// A declared (UDONCL_PUBLIC) resource needs no configured buffer.
// expected-remark@+1 {{static Unity resource lowering applied}}
module attributes {
    vrc.static_declared_public_resources = {resource_ids = array<i32: 0>, schema_version = 1 : i32},
    vrc.static_public_resource_aliases = {schema_version = 1 : i32, aliases = [
      {resource_id = 0 : i32, resource_kind = "allocation", public_access = "input", public_index = 0 : i32}]},
    vrc.static_unity_buffer_reconciliation = {schema_version = 1 : i32, buffers = []},
    vrc.static_unity_resource_lowering_readiness = {applied = true, pool_replacement_ready = true, schema_version = 1 : i32}} {
}
