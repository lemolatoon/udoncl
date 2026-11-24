// RUN: %vrc-opt --vrc-verify-onnx-session-bindings %s | %FileCheck %s

// A bounded dimension: the past is -1 on axis 1 with a largest extent of 7.
// It is bound with a constant extent of 3 in a buffer of pitch 8, which the
// 32-byte buffer holds.
// CHECK: vrc.onnx_session_bindings = [{access = "input", index = 0 : i64, resource_id = 0 : i64, tensor = "past"}]
module attributes {
  udoncl.onnx_binding = {
    inputs = [{capacity_bytes = 28 : i64, data_type = "float32", dimension_bounds = array<i64: 0, 7>, dimension_names = ["", "past"], dimensions = array<i64: 1, -1>, index = 0 : i32, logical_bytes = 28 : i64, name = "past", rank = 2 : i32}],
    outputs = []
  },
  vrc.static_device_global_cell_provenance = {cells = [
    {global = @__udon_onnx_bound_input_0, may_be_null = true, resource_ids = [0 : i32], unknown = false}
  ], schema_version = 1 : i32},
  vrc.static_device_resource_requirements = {resources = [
    {capacity_bytes = 32 : i64, memory_flags = 4 : i64, resource_id = 0 : i32}
  ], schema_version = 1 : i32}
} {
  "udonhl.global"() <{address_space = 0 : i32, alignment = 8 : i64, constant, initial_value = array<i8: 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0>, linkage = "internal", storage_type = !llvm.array<2 x i64>, sym_name = "dims"}> : () -> ()
  "udonhl.global"() <{address_space = 0 : i32, alignment = 8 : i64, constant, initial_value = array<i8: 1, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0>, linkage = "internal", storage_type = !llvm.array<2 x i64>, sym_name = "buffer_dims"}> : () -> ()
  func.func private @udononnxBindInputAt(!udonhl.ptr<0>, i32, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
  func.func private @udononnxRun(!udonhl.ptr<0>) -> i32
  func.func @Run(%session: !udonhl.ptr<0>, %memory: !udonhl.ptr<0>) attributes {udoncl.export} {
    %c0 = arith.constant 0 : i32
    %c2 = arith.constant 2 : i32
    %dims = udonhl.address_of @dims : <0>
    %buffer_dims = udonhl.address_of @buffer_dims : <0>
    %0 = func.call @udononnxBindInputAt(%session, %c0, %memory, %c2, %dims, %buffer_dims) : (!udonhl.ptr<0>, i32, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    %1 = func.call @udononnxRun(%session) : (!udonhl.ptr<0>) -> i32
    return
  }
}
