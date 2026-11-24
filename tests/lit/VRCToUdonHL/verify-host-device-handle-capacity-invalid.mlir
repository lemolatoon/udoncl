// RUN: %vrc-opt --split-input-file --verify-diagnostics --vrc-verify-host-device-handle-capacity %s

// A plan that undercounts the allocation contexts is a compile error naming
// both numbers and the plan.

// expected-error @below {{the OpenCL runtime has 1 buffer handles but the Host has 2 allocation contexts, each of which needs one (Host buffer plan build/host-buffer-plan.json)}}
module attributes {
  udoncl.mutable_handle_capacity = 1 : i64,
  vrc.host_buffer_plan = {base_capacity = 0 : i64, extra_buffer_bytes = [8 : i64], path = "build/host-buffer-plan.json", schema_version = 1 : i64, total_capacity = 1 : i64},
  vrc.static_device_resource_discovery = {contexts = [{context_id = 0 : i32}, {context_id = 1 : i32}], schema_version = 1 : i32}
} {
}

// -----

// An ONNX Host's capacity without a plan is checked the same way.

// expected-error @below {{the OpenCL runtime has 0 buffer handles but the Host has 1 allocation contexts, each of which needs one (no Host buffer plan)}}
module attributes {
  udoncl.mutable_handle_capacity = 0 : i64,
  vrc.static_device_resource_discovery = {contexts = [{context_id = 0 : i32}], schema_version = 1 : i32}
} {
}

// -----

// expected-error @below {{static Device resource discovery has no contexts}}
module attributes {
  udoncl.mutable_handle_capacity = 1 : i64,
  vrc.static_device_resource_discovery = {schema_version = 1 : i32}
} {
}
