// RUN: %vrc-opt --split-input-file --vrc-verify-onnx-session-bindings %s | %FileCheck %s

// Each graph tensor holds one Device buffer of sufficient capacity; the
// record names the UDONCL_PUBLIC variable holding it, if any.
// CHECK-LABEL: module attributes
// CHECK-SAME: vrc.onnx_session_bindings = [{access = "input", host_variable = @x_variable, index = 0 : i64, resource_id = 0 : i64, tensor = "x"}, {access = "output", index = 0 : i64, resource_id = 1 : i64, tensor = "y"}]
module attributes {
  udoncl.onnx_binding = {
    inputs = [{capacity_bytes = 16 : i64, data_type = "float32", dimensions = array<i64: 2, 2>, index = 0 : i32, logical_bytes = 16 : i64, name = "x", rank = 2 : i32}],
    outputs = [{capacity_bytes = 16 : i64, data_type = "float32", dimensions = array<i64: 2, 2>, index = 0 : i32, logical_bytes = 16 : i64, name = "y", rank = 2 : i32}]
  },
  vrc.static_device_global_cell_provenance = {cells = [
    {global = @__udon_onnx_bound_input_0, may_be_null = true, resource_ids = [0 : i32], unknown = false},
    {global = @__udon_onnx_bound_output_0, may_be_null = true, resource_ids = [1 : i32], unknown = false},
    {global = @x_variable, may_be_null = true, resource_ids = [0 : i32], unknown = false}
  ], schema_version = 1 : i32},
  vrc.static_device_resource_requirements = {resources = [
    {capacity_bytes = 16 : i64, memory_flags = 4 : i64, resource_id = 0 : i32},
    {capacity_bytes = 64 : i64, memory_flags = 2 : i64, resource_id = 1 : i32}
  ], schema_version = 1 : i32}
} {
  "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, initial_value = 0 : i32, linkage = "internal", storage_type = !udonhl.ptr<0>, sym_name = "x_variable"}> {udoncl.public = "float32"} : () -> ()
  "udonhl.global"() <{address_space = 0 : i32, alignment = 8 : i64, constant, initial_value = array<i8: 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0>, linkage = "internal", storage_type = !llvm.array<2 x i64>, sym_name = "dims"}> : () -> ()
  func.func private @udononnxBindInputAt(!udonhl.ptr<0>, i32, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
  func.func private @udononnxRun(!udonhl.ptr<0>) -> i32
  func.func @Initialize(%session: !udonhl.ptr<0>, %memory: !udonhl.ptr<0>) attributes {udoncl.export} {
    %c0 = arith.constant 0 : i32
    %c2 = arith.constant 2 : i32
    %dims = udonhl.address_of @dims : <0>
    %0 = func.call @udononnxBindInputAt(%session, %c0, %memory, %c2, %dims, %dims) : (!udonhl.ptr<0>, i32, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    %1 = func.call @udononnxRun(%session) : (!udonhl.ptr<0>) -> i32
    return
  }
}

// -----

// A Host that does not use the session API is not checked.
// CHECK-LABEL: module attributes
// CHECK-NOT: vrc.onnx_session_bindings
// CHECK-SAME: udoncl.onnx_binding
module attributes {
  udoncl.onnx_binding = {
    inputs = [{capacity_bytes = 16 : i64, data_type = "float32", dimensions = array<i64: 2, 2>, index = 0 : i32, logical_bytes = 16 : i64, name = "x", rank = 2 : i32}],
    outputs = []
  },
  vrc.onnx_session_bindings = []
} {
}
