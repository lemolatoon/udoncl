// RUN: %vrc-opt --split-input-file --vrc-reconcile-static-unity-buffers --vrc-plan-static-unity-buffer-layout %s | %FileCheck %s

// A public resource the Host source declares (UDONCL_PUBLIC) defines its
// Unity buffer itself: without a configured buffer it is neither unused nor
// a layout blocker, and the canonical layout publishes it.
// CHECK-LABEL: module attributes
// CHECK-SAME: vrc.static_unity_buffer_layout_plan = {additional_planned_count = 1 : i32, blocker_count = 0 : i32, buffers = [{buffer_id = 1 : i32, capacity_bytes = 128 : i64, data_type = "float32", format = "rgba8_unorm", heap_slot = "__udoncl_buffer_1", logical_bytes = 128 : i64, name = "image", persistent = true, public = true, public_access = "input", public_index = 0 : i32, resource_id = 0 : i32, resource_kind = "allocation", texture_id = 1 : i32}]
// CHECK-SAME: ready = true
// CHECK-SAME: vrc.static_unity_buffer_reconciliation = {buffers = [], configured_count = 0 : i32, declared_alias_count = 1 : i32, resolved_alias_count = 0 : i32, schema_version = 1 : i32, unresolved_private_count = 0 : i32, unresolved_public_count = 0 : i32, unused_alias_count = 0 : i32}
module attributes {
    vrc.static_declared_public_resources = {resource_ids = array<i32: 0>, schema_version = 1 : i32},
    vrc.static_device_resource_inventory = {schema_version = 1 : i32, resources = [
      {capacity_bytes = 128 : i64, data_type = "float32", format = "rgba8_unorm", logical_bytes = 128 : i64, name = "image", persistent = true, public = true, public_access = "input", resource_id = 0 : i32, resource_kind = "allocation"}]},
    vrc.unity_texture_assignments = {schema_version = 1 : i32, assignments = [
      {capacity_bytes = 128 : i64, format = "rgba8_unorm", persistent = true, public = true, public_access = "input", resource_id = 0 : i32, resource_kind = "allocation", texture_id = 1 : i32}]},
    vrc.static_public_resource_aliases = {schema_version = 1 : i32, resolved_count = 1 : i32, aliases = [
      {public_access = "input", public_index = 0 : i32, resource_id = 0 : i32, resource_kind = "allocation", texture_id = 1 : i32}]},
    vrc.unity_configured_buffers = {schema_version = 1 : i32, buffers = []}} {
}

// -----

// Any other public resource still needs a configured buffer.
// CHECK-LABEL: module attributes
// CHECK-SAME: blocker_count = 1 : i32
// CHECK-SAME: ready = false
// CHECK-SAME: vrc.static_unity_buffer_reconciliation = {buffers = [], configured_count = 0 : i32, resolved_alias_count = 0 : i32, schema_version = 1 : i32, unresolved_private_count = 0 : i32, unresolved_public_count = 0 : i32, unused_alias_count = 1 : i32}
module attributes {
    vrc.static_device_resource_inventory = {schema_version = 1 : i32, resources = [
      {capacity_bytes = 128 : i64, data_type = "float32", format = "rgba8_unorm", logical_bytes = 128 : i64, name = "image", persistent = true, public = true, public_access = "input", resource_id = 0 : i32, resource_kind = "allocation"}]},
    vrc.unity_texture_assignments = {schema_version = 1 : i32, assignments = [
      {capacity_bytes = 128 : i64, format = "rgba8_unorm", persistent = true, public = true, public_access = "input", resource_id = 0 : i32, resource_kind = "allocation", texture_id = 1 : i32}]},
    vrc.static_public_resource_aliases = {schema_version = 1 : i32, resolved_count = 1 : i32, aliases = [
      {public_access = "input", public_index = 0 : i32, resource_id = 0 : i32, resource_kind = "allocation", texture_id = 1 : i32}]},
    vrc.unity_configured_buffers = {schema_version = 1 : i32, buffers = []}} {
}
