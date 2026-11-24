// RUN: %vrc-opt --split-input-file --verify-diagnostics --vrc-verify-onnx-session-bindings %s

// A Host that runs the graph binds every tensor.
// expected-error @below {{ONNX graph output "y" is never bound; bind a Device buffer to it with udononnxBindOutput before udononnxRun}}
module attributes {
  udoncl.onnx_binding = {
    inputs = [],
    outputs = [{capacity_bytes = 16 : i64, data_type = "float32", dimensions = array<i64: 4>, index = 0 : i32, logical_bytes = 16 : i64, name = "y", rank = 1 : i32}]
  }
} {
  func.func private @udononnxRun(!udonhl.ptr<0>) -> i32
  func.func @Run(%session: !udonhl.ptr<0>) attributes {udoncl.export} {
    %0 = func.call @udononnxRun(%session) : (!udonhl.ptr<0>) -> i32
    return
  }
}

// -----

// A tensor holds exactly one Device buffer.
// expected-error @below {{ONNX graph input "x" is bound to the Device buffers of 2 allocation contexts; bind exactly one buffer}}
module attributes {
  udoncl.onnx_binding = {
    inputs = [{capacity_bytes = 16 : i64, data_type = "float32", dimensions = array<i64: 4>, index = 0 : i32, logical_bytes = 16 : i64, name = "x", rank = 1 : i32}],
    outputs = []
  },
  vrc.static_device_global_cell_provenance = {cells = [
    {global = @__udon_onnx_bound_input_0, may_be_null = true, resource_ids = [0 : i32, 1 : i32], unknown = false}
  ], schema_version = 1 : i32}
} {
  func.func private @udononnxRun(!udonhl.ptr<0>) -> i32
  func.func @Run(%session: !udonhl.ptr<0>) attributes {udoncl.export} {
    %0 = func.call @udononnxRun(%session) : (!udonhl.ptr<0>) -> i32
    return
  }
}

// -----

// Each call binding a tensor to more than one buffer is named: the export
// making it (or the function and the exports it is reached from) and the
// allocation of the bound memory, when it is known.
// expected-error @below {{ONNX graph input "x" is bound to the Device buffers of 2 allocation contexts; bind exactly one buffer}}
module attributes {
  udoncl.onnx_binding = {
    inputs = [{capacity_bytes = 16 : i64, data_type = "float32", dimensions = array<i64: 4>, index = 0 : i32, logical_bytes = 16 : i64, name = "x", rank = 1 : i32}],
    outputs = []
  },
  vrc.static_device_global_cell_provenance = {cells = [
    {global = @__udon_onnx_bound_input_0, may_be_null = true, resource_ids = [0 : i32, 1 : i32], unknown = false},
    {global = @b, may_be_null = true, resource_ids = [1 : i32], unknown = false}
  ], schema_version = 1 : i32},
  vrc.static_device_resource_discovery = {contexts = [
    {allocation_site = {function = @create, occurrence = 0 : i32, primitive = "clCreateBuffer", resource_result = 0 : i32}, call_path = [], context_id = 0 : i32, root = @Initialize},
    {allocation_site = {function = @create, occurrence = 0 : i32, primitive = "clCreateBuffer", resource_result = 0 : i32}, call_path = [], context_id = 1 : i32, root = @Initialize}
  ], schema_version = 1 : i32}
} {
  "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, initial_value = 0 : i32, linkage = "internal", storage_type = !udonhl.ptr<0>, sym_name = "b"}> {udoncl.public = "float32"} : () -> ()
  "udonhl.global"() <{address_space = 0 : i32, alignment = 8 : i64, constant, initial_value = array<i8: 4, 0, 0, 0, 0, 0, 0, 0>, linkage = "internal", storage_type = !llvm.array<1 x i64>, sym_name = "dims"}> : () -> ()
  func.func private @udononnxBindInputAt(!udonhl.ptr<0>, i32, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
  func.func private @udononnxRun(!udonhl.ptr<0>) -> i32
  func.func @Initialize(%session: !udonhl.ptr<0>, %memory: !udonhl.ptr<0>) attributes {udoncl.export} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %dims = udonhl.address_of @dims : <0>
    // expected-note @below {{bound here in export @Initialize, to the Device buffer of allocation context 0 (clCreateBuffer in @create)}}
    %0 = func.call @udononnxBindInputAt(%session, %c0, %memory, %c1, %dims, %dims) {vrc.static_device_operand_provenance = [{cases = [{call_path = [], may_be_null = false, resource_ids = [0 : i32], root = @Initialize, unknown = false}], index = 2 : i32}]} : (!udonhl.ptr<0>, i32, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    return
  }
  func.func private @rebind(%session: !udonhl.ptr<0>, %memory: !udonhl.ptr<0>) -> i32 {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %dims = udonhl.address_of @dims : <0>
    // expected-note @below {{bound here in @rebind, reached from export @Rebind, to the Device buffer of allocation context 1 (clCreateBuffer in @create, held by UDONCL_PUBLIC variable b)}}
    %0 = func.call @udononnxBindInputAt(%session, %c0, %memory, %c1, %dims, %dims) {vrc.static_device_operand_provenance = [{cases = [{call_path = [], may_be_null = false, resource_ids = [1 : i32], root = @Rebind, unknown = false}], index = 2 : i32}]} : (!udonhl.ptr<0>, i32, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    return %0 : i32
  }
  func.func @Rebind(%session: !udonhl.ptr<0>, %memory: !udonhl.ptr<0>) attributes {udoncl.export} {
    %0 = func.call @rebind(%session, %memory) : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    %1 = func.call @udononnxRun(%session) : (!udonhl.ptr<0>) -> i32
    return
  }
  func.func @Unknown(%session: !udonhl.ptr<0>, %memory: !udonhl.ptr<0>) attributes {udoncl.export} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %dims = udonhl.address_of @dims : <0>
    // expected-note @below {{bound here in export @Unknown}}
    %0 = func.call @udononnxBindInputAt(%session, %c0, %memory, %c1, %dims, %dims) : (!udonhl.ptr<0>, i32, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    return
  }
}

// -----

// A buffer of unknown origin cannot be assigned at compile time.
// expected-error @below {{the Device buffer bound to ONNX graph input "x" may be of unknown origin}}
module attributes {
  udoncl.onnx_binding = {
    inputs = [{capacity_bytes = 16 : i64, data_type = "float32", dimensions = array<i64: 4>, index = 0 : i32, logical_bytes = 16 : i64, name = "x", rank = 1 : i32}],
    outputs = []
  },
  vrc.static_device_global_cell_provenance = {cells = [
    {global = @__udon_onnx_bound_input_0, may_be_null = true, resource_ids = [0 : i32], unknown = true}
  ], schema_version = 1 : i32}
} {
  func.func private @udononnxRun(!udonhl.ptr<0>) -> i32
  func.func @Run(%session: !udonhl.ptr<0>) attributes {udoncl.export} {
    %0 = func.call @udononnxRun(%session) : (!udonhl.ptr<0>) -> i32
    return
  }
}

// -----

// The buffer holds the whole tensor.
// expected-error @below {{ONNX graph input "x" needs 16 bytes, but its Device buffer has 8}}
module attributes {
  udoncl.onnx_binding = {
    inputs = [{capacity_bytes = 16 : i64, data_type = "float32", dimensions = array<i64: 4>, index = 0 : i32, logical_bytes = 16 : i64, name = "x", rank = 1 : i32}],
    outputs = []
  },
  vrc.static_device_global_cell_provenance = {cells = [
    {global = @__udon_onnx_bound_input_0, may_be_null = true, resource_ids = [0 : i32], unknown = false}
  ], schema_version = 1 : i32},
  vrc.static_device_resource_requirements = {resources = [
    {capacity_bytes = 8 : i64, memory_flags = 4 : i64, resource_id = 0 : i32}
  ], schema_version = 1 : i32}
} {
  func.func private @udononnxRun(!udonhl.ptr<0>) -> i32
  func.func @Run(%session: !udonhl.ptr<0>) attributes {udoncl.export} {
    %0 = func.call @udononnxRun(%session) : (!udonhl.ptr<0>) -> i32
    return
  }
}

// -----

// A UDONCL_PUBLIC variable bound to a tensor has its element type.
module attributes {
  udoncl.onnx_binding = {
    inputs = [{capacity_bytes = 16 : i64, data_type = "float32", dimensions = array<i64: 4>, index = 0 : i32, logical_bytes = 16 : i64, name = "x", rank = 1 : i32}],
    outputs = []
  },
  vrc.static_device_global_cell_provenance = {cells = [
    {global = @__udon_onnx_bound_input_0, may_be_null = true, resource_ids = [0 : i32], unknown = false},
    {global = @ids, may_be_null = true, resource_ids = [0 : i32], unknown = false}
  ], schema_version = 1 : i32},
  vrc.static_device_resource_requirements = {resources = [
    {capacity_bytes = 16 : i64, memory_flags = 4 : i64, resource_id = 0 : i32}
  ], schema_version = 1 : i32}
} {
  // expected-error @below {{UDONCL_PUBLIC variable ids holds int32 elements, but ONNX graph input "x" bound to it is float32}}
  "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, initial_value = 0 : i32, linkage = "internal", storage_type = !udonhl.ptr<0>, sym_name = "ids"}> {udoncl.public = "int32"} : () -> ()
  func.func private @udononnxRun(!udonhl.ptr<0>) -> i32
  func.func @Run(%session: !udonhl.ptr<0>) attributes {udoncl.export} {
    %0 = func.call @udononnxRun(%session) : (!udonhl.ptr<0>) -> i32
    return
  }
}

// -----

// A constant rank and constant dimensions are the model's.
module attributes {
  udoncl.onnx_binding = {
    inputs = [{capacity_bytes = 16 : i64, data_type = "float32", dimensions = array<i64: 2, 2>, index = 0 : i32, logical_bytes = 16 : i64, name = "x", rank = 2 : i32}],
    outputs = []
  }
} {
  "udonhl.global"() <{address_space = 0 : i32, alignment = 8 : i64, constant, initial_value = array<i8: 1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0>, linkage = "internal", storage_type = !llvm.array<2 x i64>, sym_name = "dims"}> : () -> ()
  func.func private @udononnxBindInputAt(!udonhl.ptr<0>, i32, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
  func.func @Bind(%session: !udonhl.ptr<0>, %memory: !udonhl.ptr<0>) attributes {udoncl.export} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %dims = udonhl.address_of @dims : <0>
    // expected-error @below {{ONNX graph input "x" has rank 2, bound with rank 1}}
    %0 = func.call @udononnxBindInputAt(%session, %c0, %memory, %c1, %dims, %dims) : (!udonhl.ptr<0>, i32, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    // expected-error @below {{ONNX graph input "x" has dimensions [2, 2], bound with [1, 4]}}
    %1 = func.call @udononnxBindInputAt(%session, %c0, %memory, %c2, %dims, %dims) : (!udonhl.ptr<0>, i32, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    return
  }
}
