// RUN: %vrc-opt --split-input-file --verify-diagnostics --vrc-verify-onnx-session-bindings %s

// A constant extent beyond the bound of the dimension.
module attributes {
  udoncl.onnx_binding = {
    inputs = [{capacity_bytes = 28 : i64, data_type = "float32", dimension_bounds = array<i64: 0, 7>, dimension_names = ["", "past"], dimensions = array<i64: 1, -1>, index = 0 : i32, logical_bytes = 28 : i64, name = "past", rank = 2 : i32}],
    outputs = []
  }
} {
  "udonhl.global"() <{address_space = 0 : i32, alignment = 8 : i64, constant, initial_value = array<i8: 1, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0>, linkage = "internal", storage_type = !llvm.array<2 x i64>, sym_name = "dims"}> : () -> ()
  func.func private @udononnxBindInputAt(!udonhl.ptr<0>, i32, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
  func.func @Bind(%session: !udonhl.ptr<0>, %memory: !udonhl.ptr<0>, %none: !udonhl.ptr<0>) attributes {udoncl.export} {
    %c0 = arith.constant 0 : i32
    %c2 = arith.constant 2 : i32
    %dims = udonhl.address_of @dims : <0>
    // expected-error @below {{ONNX graph input "past" has dimensions [1, -1], bound with [1, 9]; a -1 dimension takes an extent up to [0, 7]}}
    %0 = func.call @udononnxBindInputAt(%session, %c0, %memory, %c2, %dims, %none) : (!udonhl.ptr<0>, i32, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    return
  }
}

// -----

// A constant buffer_dims smaller than the tensor on the dynamic axis.
module attributes {
  udoncl.onnx_binding = {
    inputs = [{capacity_bytes = 28 : i64, data_type = "float32", dimension_bounds = array<i64: 0, 7>, dimension_names = ["", "past"], dimensions = array<i64: 1, -1>, index = 0 : i32, logical_bytes = 28 : i64, name = "past", rank = 2 : i32}],
    outputs = []
  }
} {
  "udonhl.global"() <{address_space = 0 : i32, alignment = 8 : i64, constant, initial_value = array<i8: 1, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0>, linkage = "internal", storage_type = !llvm.array<2 x i64>, sym_name = "dims"}> : () -> ()
  "udonhl.global"() <{address_space = 0 : i32, alignment = 8 : i64, constant, initial_value = array<i8: 1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0>, linkage = "internal", storage_type = !llvm.array<2 x i64>, sym_name = "buffer_dims"}> : () -> ()
  func.func private @udononnxBindInputAt(!udonhl.ptr<0>, i32, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
  func.func @Bind(%session: !udonhl.ptr<0>, %memory: !udonhl.ptr<0>) attributes {udoncl.export} {
    %c0 = arith.constant 0 : i32
    %c2 = arith.constant 2 : i32
    %dims = udonhl.address_of @dims : <0>
    %buffer_dims = udonhl.address_of @buffer_dims : <0>
    // expected-error @below {{ONNX graph input "past" is bound with buffer_dims [1, 4], which must equal its static dimensions and hold its dimensions on the dynamic axis}}
    %0 = func.call @udononnxBindInputAt(%session, %c0, %memory, %c2, %dims, %buffer_dims) : (!udonhl.ptr<0>, i32, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    return
  }
}

// -----

// A buffer too small for the layout it is bound with: the pitch of 16 needs
// 64 bytes, though the tensor at its bound needs 28.
// expected-error @below {{ONNX graph input "past" needs 64 bytes, but its Device buffer has 32}}
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
  "udonhl.global"() <{address_space = 0 : i32, alignment = 8 : i64, constant, initial_value = array<i8: 1, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0>, linkage = "internal", storage_type = !llvm.array<2 x i64>, sym_name = "dims"}> : () -> ()
  "udonhl.global"() <{address_space = 0 : i32, alignment = 8 : i64, constant, initial_value = array<i8: 1, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0>, linkage = "internal", storage_type = !llvm.array<2 x i64>, sym_name = "buffer_dims"}> : () -> ()
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
