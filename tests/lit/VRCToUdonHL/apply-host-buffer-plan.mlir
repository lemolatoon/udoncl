// RUN: %vrc-opt --split-input-file --vrc-apply-host-buffer-plan %s | %FileCheck %s

// The C Host's buffers follow the ONNX Host's: the handle capacity becomes
// the plan's total and every extra buffer adds its byte size. The plan stays
// on the module.

// CHECK-LABEL: module attributes
// CHECK-SAME: udoncl.mutable_buffer_bytes = array<i64: 32, 16, 8, 3136, 40>
// CHECK-SAME: udoncl.mutable_handle_capacity = 5 : i64
// CHECK-SAME: vrc.host_buffer_plan = {base_capacity = 3 : i64
module attributes {
  udoncl.mutable_buffer_bytes = array<i64: 32, 16, 8>,
  udoncl.mutable_handle_capacity = 3 : i64,
  vrc.host_buffer_plan = {base_capacity = 3 : i64,
                          extra_buffer_bytes = [3136 : i64, 40 : i64],
                          path = "host-buffer-plan.json",
                          schema_version = 1 : i64, total_capacity = 5 : i64}
} {
}

// -----

// A plan without extra buffers keeps the ONNX capacity.

// CHECK-LABEL: module attributes
// CHECK-SAME: udoncl.mutable_buffer_bytes = array<i64: 64>
// CHECK-SAME: udoncl.mutable_handle_capacity = 1 : i64
module attributes {
  udoncl.mutable_buffer_bytes = array<i64: 64>,
  udoncl.mutable_handle_capacity = 1 : i64,
  vrc.host_buffer_plan = {base_capacity = 1 : i64, extra_buffer_bytes = [],
                          path = "plan.json", schema_version = 1 : i64,
                          total_capacity = 1 : i64}
} {
}

// -----

// Without an ONNX Host the plan's base is 0: the handles are the C Host's
// buffers alone.

// CHECK-LABEL: module attributes
// CHECK-SAME: udoncl.mutable_buffer_bytes = array<i64: 128, 128, 64>
// CHECK-SAME: udoncl.mutable_handle_capacity = 3 : i64
module attributes {
  vrc.host_buffer_plan = {base_capacity = 0 : i64,
                          extra_buffer_bytes = [128 : i64, 128 : i64, 64 : i64],
                          path = "plan.json", schema_version = 1 : i64,
                          total_capacity = 3 : i64}
} {
}

// -----

// A module without a plan is unchanged.

// CHECK-LABEL: module attributes
// CHECK-SAME: udoncl.mutable_buffer_bytes = array<i64: 4>
// CHECK-SAME: udoncl.mutable_handle_capacity = 1 : i64
// CHECK-NOT: vrc.host_buffer_plan
module attributes {
  udoncl.mutable_buffer_bytes = array<i64: 4>,
  udoncl.mutable_handle_capacity = 1 : i64
} {
}
