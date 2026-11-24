// RUN: %vrc-opt --split-input-file --verify-diagnostics --vrc-plan-host-device-handles %s

// The pass needs the requirements of the contexts.

// expected-error @below {{the Host device handle plan requires static allocation contexts and their requirements}}
module attributes {
  vrc.static_device_resource_discovery = {contexts = [], schema_version = 1 : i32}
} {
}

// -----

// Every context has an inferred capacity.

// expected-error @below {{static allocation context has no inferred capacity}}
module attributes {
  vrc.static_device_resource_discovery = {contexts = [{context_id = 0 : i32}], schema_version = 1 : i32},
  vrc.static_device_resource_requirements = {resources = [], schema_version = 1 : i32}
} {
}
