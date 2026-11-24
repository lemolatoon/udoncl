// RUN: %vrc-opt --split-input-file --verify-diagnostics --vrc-verify-onnx-session-bindings %s

// Only the first operand of an in-place Concat and its result share a
// buffer, a CL_MEM_READ_WRITE one, with the same layout.

// The step input is not part of the pair.
// expected-error @below {{ONNX graph input "past", input "step" and output "present" are bound to one Device buffer; only the first operand of an in-place Concat and its result may share a buffer}}
module attributes {
  udoncl.onnx_binding = {
    in_place_concats = [{axis = 1 : i32, input = 0 : i32, output = 0 : i32}],
    inputs = [{capacity_bytes = 28 : i64, data_type = "float32", dimension_bounds = array<i64: 0, 7>, dimension_names = ["", "past"], dimensions = array<i64: 1, -1>, index = 0 : i32, logical_bytes = 28 : i64, name = "past", rank = 2 : i32}, {capacity_bytes = 4 : i64, data_type = "float32", dimensions = array<i64: 1, 1>, index = 1 : i32, logical_bytes = 4 : i64, name = "step", rank = 2 : i32}],
    outputs = [{capacity_bytes = 32 : i64, data_type = "float32", dimension_bounds = array<i64: 0, 8>, dimension_names = ["", "past"], dimensions = array<i64: 1, -1>, index = 0 : i32, logical_bytes = 32 : i64, name = "present", rank = 2 : i32}]
  },
  vrc.static_device_global_cell_provenance = {cells = [
    {global = @__udon_onnx_bound_input_0, may_be_null = true, resource_ids = [0 : i32], unknown = false},
    {global = @__udon_onnx_bound_input_1, may_be_null = true, resource_ids = [0 : i32], unknown = false},
    {global = @__udon_onnx_bound_output_0, may_be_null = true, resource_ids = [0 : i32], unknown = false}
  ], schema_version = 1 : i32},
  vrc.static_device_resource_requirements = {resources = [
    {capacity_bytes = 32 : i64, memory_flags = 1 : i64, resource_id = 0 : i32},
    {capacity_bytes = 4 : i64, memory_flags = 1 : i64, resource_id = 1 : i32}
  ], schema_version = 1 : i32}
} {
  "udonhl.global"() <{address_space = 0 : i32, alignment = 8 : i64, constant, initial_value = array<i8: 1, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0>, linkage = "internal", storage_type = !llvm.array<2 x i64>, sym_name = "buffer_dims"}> : () -> ()
  "udonhl.global"() <{address_space = 0 : i32, alignment = 8 : i64, constant, initial_value = array<i8: 1, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0>, linkage = "internal", storage_type = !llvm.array<2 x i64>, sym_name = "past_buffer_dims"}> : () -> ()
  func.func private @udononnxBindInputAt(!udonhl.ptr<0>, i32, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
  func.func private @udononnxBindOutputAt(!udonhl.ptr<0>, i32, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
  func.func private @udononnxRun(!udonhl.ptr<0>) -> i32
  func.func @Run(%session: !udonhl.ptr<0>, %kv: !udonhl.ptr<0>, %step: !udonhl.ptr<0>, %past_dims: !udonhl.ptr<0>, %present_dims: !udonhl.ptr<0>, %none: !udonhl.ptr<0>) attributes {udoncl.export} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %buffer_dims = udonhl.address_of @buffer_dims : <0>
    %past_buffer_dims = udonhl.address_of @past_buffer_dims : <0>
    %0 = func.call @udononnxBindInputAt(%session, %c0, %kv, %c2, %past_dims, %past_buffer_dims) : (!udonhl.ptr<0>, i32, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    %1 = func.call @udononnxBindInputAt(%session, %c1, %step, %c2, %none, %none) : (!udonhl.ptr<0>, i32, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    %2 = func.call @udononnxBindOutputAt(%session, %c0, %kv, %c2, %present_dims, %buffer_dims) : (!udonhl.ptr<0>, i32, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    %3 = func.call @udononnxRun(%session) : (!udonhl.ptr<0>) -> i32
    return
  }
}

// -----

// A read-only buffer cannot hold the result.
// expected-error @below {{the Device buffer ONNX graph input "past" and output "present" share must be CL_MEM_READ_WRITE}}
module attributes {
  udoncl.onnx_binding = {
    in_place_concats = [{axis = 1 : i32, input = 0 : i32, output = 0 : i32}],
    inputs = [{capacity_bytes = 28 : i64, data_type = "float32", dimension_bounds = array<i64: 0, 7>, dimension_names = ["", "past"], dimensions = array<i64: 1, -1>, index = 0 : i32, logical_bytes = 28 : i64, name = "past", rank = 2 : i32}, {capacity_bytes = 4 : i64, data_type = "float32", dimensions = array<i64: 1, 1>, index = 1 : i32, logical_bytes = 4 : i64, name = "step", rank = 2 : i32}],
    outputs = [{capacity_bytes = 32 : i64, data_type = "float32", dimension_bounds = array<i64: 0, 8>, dimension_names = ["", "past"], dimensions = array<i64: 1, -1>, index = 0 : i32, logical_bytes = 32 : i64, name = "present", rank = 2 : i32}]
  },
  vrc.static_device_global_cell_provenance = {cells = [
    {global = @__udon_onnx_bound_input_0, may_be_null = true, resource_ids = [0 : i32], unknown = false},
    {global = @__udon_onnx_bound_input_1, may_be_null = true, resource_ids = [1 : i32], unknown = false},
    {global = @__udon_onnx_bound_output_0, may_be_null = true, resource_ids = [0 : i32], unknown = false}
  ], schema_version = 1 : i32},
  vrc.static_device_resource_requirements = {resources = [
    {capacity_bytes = 32 : i64, memory_flags = 4 : i64, resource_id = 0 : i32},
    {capacity_bytes = 4 : i64, memory_flags = 1 : i64, resource_id = 1 : i32}
  ], schema_version = 1 : i32}
} {
  "udonhl.global"() <{address_space = 0 : i32, alignment = 8 : i64, constant, initial_value = array<i8: 1, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0>, linkage = "internal", storage_type = !llvm.array<2 x i64>, sym_name = "buffer_dims"}> : () -> ()
  "udonhl.global"() <{address_space = 0 : i32, alignment = 8 : i64, constant, initial_value = array<i8: 1, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0>, linkage = "internal", storage_type = !llvm.array<2 x i64>, sym_name = "past_buffer_dims"}> : () -> ()
  func.func private @udononnxBindInputAt(!udonhl.ptr<0>, i32, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
  func.func private @udononnxBindOutputAt(!udonhl.ptr<0>, i32, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
  func.func private @udononnxRun(!udonhl.ptr<0>) -> i32
  func.func @Run(%session: !udonhl.ptr<0>, %kv: !udonhl.ptr<0>, %step: !udonhl.ptr<0>, %past_dims: !udonhl.ptr<0>, %present_dims: !udonhl.ptr<0>, %none: !udonhl.ptr<0>) attributes {udoncl.export} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %buffer_dims = udonhl.address_of @buffer_dims : <0>
    %past_buffer_dims = udonhl.address_of @past_buffer_dims : <0>
    %0 = func.call @udononnxBindInputAt(%session, %c0, %kv, %c2, %past_dims, %past_buffer_dims) : (!udonhl.ptr<0>, i32, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    %1 = func.call @udononnxBindInputAt(%session, %c1, %step, %c2, %none, %none) : (!udonhl.ptr<0>, i32, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    %2 = func.call @udononnxBindOutputAt(%session, %c0, %kv, %c2, %present_dims, %buffer_dims) : (!udonhl.ptr<0>, i32, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    %3 = func.call @udononnxRun(%session) : (!udonhl.ptr<0>) -> i32
    return
  }
}

// -----

// The past is not the leading block of the present's layout.
// expected-error @below {{ONNX graph input "past" and output "present" share a Device buffer but are bound with buffer_dims [1, 7] and [1, 8]; the input must be the leading block of the output's layout}}
module attributes {
  udoncl.onnx_binding = {
    in_place_concats = [{axis = 1 : i32, input = 0 : i32, output = 0 : i32}],
    inputs = [{capacity_bytes = 28 : i64, data_type = "float32", dimension_bounds = array<i64: 0, 7>, dimension_names = ["", "past"], dimensions = array<i64: 1, -1>, index = 0 : i32, logical_bytes = 28 : i64, name = "past", rank = 2 : i32}, {capacity_bytes = 4 : i64, data_type = "float32", dimensions = array<i64: 1, 1>, index = 1 : i32, logical_bytes = 4 : i64, name = "step", rank = 2 : i32}],
    outputs = [{capacity_bytes = 32 : i64, data_type = "float32", dimension_bounds = array<i64: 0, 8>, dimension_names = ["", "past"], dimensions = array<i64: 1, -1>, index = 0 : i32, logical_bytes = 32 : i64, name = "present", rank = 2 : i32}]
  },
  vrc.static_device_global_cell_provenance = {cells = [
    {global = @__udon_onnx_bound_input_0, may_be_null = true, resource_ids = [0 : i32], unknown = false},
    {global = @__udon_onnx_bound_input_1, may_be_null = true, resource_ids = [1 : i32], unknown = false},
    {global = @__udon_onnx_bound_output_0, may_be_null = true, resource_ids = [0 : i32], unknown = false}
  ], schema_version = 1 : i32},
  vrc.static_device_resource_requirements = {resources = [
    {capacity_bytes = 32 : i64, memory_flags = 1 : i64, resource_id = 0 : i32},
    {capacity_bytes = 4 : i64, memory_flags = 1 : i64, resource_id = 1 : i32}
  ], schema_version = 1 : i32}
} {
  "udonhl.global"() <{address_space = 0 : i32, alignment = 8 : i64, constant, initial_value = array<i8: 1, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0>, linkage = "internal", storage_type = !llvm.array<2 x i64>, sym_name = "buffer_dims"}> : () -> ()
  "udonhl.global"() <{address_space = 0 : i32, alignment = 8 : i64, constant, initial_value = array<i8: 1, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0>, linkage = "internal", storage_type = !llvm.array<2 x i64>, sym_name = "past_buffer_dims"}> : () -> ()
  func.func private @udononnxBindInputAt(!udonhl.ptr<0>, i32, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
  func.func private @udononnxBindOutputAt(!udonhl.ptr<0>, i32, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
  func.func private @udononnxRun(!udonhl.ptr<0>) -> i32
  func.func @Run(%session: !udonhl.ptr<0>, %kv: !udonhl.ptr<0>, %step: !udonhl.ptr<0>, %past_dims: !udonhl.ptr<0>, %present_dims: !udonhl.ptr<0>, %none: !udonhl.ptr<0>) attributes {udoncl.export} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %buffer_dims = udonhl.address_of @buffer_dims : <0>
    %past_buffer_dims = udonhl.address_of @past_buffer_dims : <0>
    %0 = func.call @udononnxBindInputAt(%session, %c0, %kv, %c2, %past_dims, %past_buffer_dims) : (!udonhl.ptr<0>, i32, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    %1 = func.call @udononnxBindInputAt(%session, %c1, %step, %c2, %none, %none) : (!udonhl.ptr<0>, i32, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    %2 = func.call @udononnxBindOutputAt(%session, %c0, %kv, %c2, %present_dims, %buffer_dims) : (!udonhl.ptr<0>, i32, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    %3 = func.call @udononnxRun(%session) : (!udonhl.ptr<0>) -> i32
    return
  }
}
