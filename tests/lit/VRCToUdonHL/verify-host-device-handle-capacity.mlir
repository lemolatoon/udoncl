// RUN: %vrc-opt --split-input-file --vrc-verify-host-device-handle-capacity %s | %FileCheck %s

// A runtime with one handle per allocation context, or more, is accepted.

// CHECK-LABEL: module attributes
// CHECK-SAME: udoncl.mutable_handle_capacity = 2 : i64
module attributes {
  udoncl.mutable_handle_capacity = 2 : i64,
  vrc.static_device_resource_discovery = {contexts = [{context_id = 0 : i32}, {context_id = 1 : i32}], schema_version = 1 : i32}
} {
}

// -----

// CHECK-LABEL: module attributes
// CHECK-SAME: udoncl.mutable_handle_capacity = 3 : i64
module attributes {
  udoncl.mutable_handle_capacity = 3 : i64,
  vrc.static_device_resource_discovery = {contexts = [{context_id = 0 : i32}], schema_version = 1 : i32}
} {
}

// -----

// Without a planned capacity, or without discovered contexts, there is
// nothing to check.

// CHECK-LABEL: module attributes
// CHECK-SAME: vrc.static_device_resource_discovery
module attributes {
  vrc.static_device_resource_discovery = {contexts = [{context_id = 0 : i32}], schema_version = 1 : i32}
} {
}

// -----

// CHECK-LABEL: module attributes
// CHECK-SAME: udoncl.mutable_handle_capacity = 0 : i64
module attributes {
  udoncl.mutable_handle_capacity = 0 : i64
} {
}
