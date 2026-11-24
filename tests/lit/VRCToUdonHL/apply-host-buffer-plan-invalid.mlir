// RUN: %vrc-opt --split-input-file --verify-diagnostics --vrc-apply-host-buffer-plan %s

// The plan's base is the ONNX Host's capacity.

// expected-error @below {{Host buffer plan is inconsistent with ONNX capacity}}
module attributes {
  udoncl.mutable_buffer_bytes = array<i64: 32, 16>,
  udoncl.mutable_handle_capacity = 2 : i64,
  vrc.host_buffer_plan = {base_capacity = 3 : i64, extra_buffer_bytes = [],
                          path = "plan.json", schema_version = 1 : i64,
                          total_capacity = 3 : i64}
} {
}

// -----

// The total is the base plus the extra buffers.

// expected-error @below {{Host buffer plan is inconsistent with ONNX capacity}}
module attributes {
  udoncl.mutable_buffer_bytes = array<i64: 32>,
  udoncl.mutable_handle_capacity = 1 : i64,
  vrc.host_buffer_plan = {base_capacity = 1 : i64,
                          extra_buffer_bytes = [8 : i64],
                          path = "plan.json", schema_version = 1 : i64,
                          total_capacity = 3 : i64}
} {
}

// -----

// Only schema version 1 is known.

// expected-error @below {{Host buffer plan is inconsistent with ONNX capacity}}
module attributes {
  udoncl.mutable_buffer_bytes = array<i64: 32>,
  udoncl.mutable_handle_capacity = 1 : i64,
  vrc.host_buffer_plan = {base_capacity = 1 : i64, extra_buffer_bytes = [],
                          path = "plan.json", schema_version = 2 : i64,
                          total_capacity = 1 : i64}
} {
}

// -----

// A plan with a base needs the ONNX capacity it extends.

// expected-error @below {{Host buffer plan is inconsistent with ONNX capacity}}
module attributes {
  vrc.host_buffer_plan = {base_capacity = 1 : i64,
                          extra_buffer_bytes = [8 : i64],
                          path = "plan.json", schema_version = 1 : i64,
                          total_capacity = 2 : i64}
} {
}

// -----

// Every extra buffer has a positive byte size.

// expected-error @below {{Host buffer plan contains an invalid byte size}}
module attributes {
  udoncl.mutable_buffer_bytes = array<i64: 32>,
  udoncl.mutable_handle_capacity = 1 : i64,
  vrc.host_buffer_plan = {base_capacity = 1 : i64,
                          extra_buffer_bytes = [8 : i64, 0 : i64],
                          path = "plan.json", schema_version = 1 : i64,
                          total_capacity = 3 : i64}
} {
}

// -----

// An extra byte size that is not an integer is invalid.

// expected-error @below {{Host buffer plan contains an invalid byte size}}
module attributes {
  udoncl.mutable_buffer_bytes = array<i64: 32>,
  udoncl.mutable_handle_capacity = 1 : i64,
  vrc.host_buffer_plan = {base_capacity = 1 : i64,
                          extra_buffer_bytes = [unit],
                          path = "plan.json", schema_version = 1 : i64,
                          total_capacity = 2 : i64}
} {
}
