// RUN: %vrc-opt --split-input-file --verify-diagnostics "--vrc-derive-onnx-unity-buffers=public-tensor-buffers=true" %s

// Public tensor buffers require an ONNX Host.

// expected-error @below {{--onnx-public-tensor-buffers (ONNX_PUBLIC_TENSOR_BUFFERS) requires an ONNX Host}}
module {
}

// -----

// The ONNX binding lists its inputs and outputs.

// expected-error @below {{ONNX binding has no input/output metadata}}
module attributes {
  udoncl.mutable_buffer_bytes = array<i64: 4>,
  udoncl.mutable_handle_capacity = 1 : i64,
  udoncl.onnx_binding = {inputs = []}
} {
}

// -----

// A public tensor is a float32, int32 or int64 tensor whose logical size fits
// its capacity.

// expected-error @below {{ONNX public tensor has an unsupported binding ABI}}
module attributes {
  udoncl.mutable_buffer_bytes = array<i64: 4>,
  udoncl.mutable_handle_capacity = 1 : i64,
  udoncl.onnx_binding = {
    inputs = [{capacity_bytes = 4 : i64, data_type = "float16", index = 0 : i32, logical_bytes = 4 : i64, name = "x"}],
    outputs = []
  }
} {
}

// -----

// Two tensors whose C# names and suffixed names collide cannot be named.

// expected-error @below {{ONNX public tensor names are not uniquely mappable}}
module attributes {
  udoncl.mutable_buffer_bytes = array<i64: 4, 4, 4>,
  udoncl.mutable_handle_capacity = 3 : i64,
  udoncl.onnx_binding = {
    inputs = [
      {capacity_bytes = 4 : i64, data_type = "float32", index = 0 : i32, logical_bytes = 4 : i64, name = "x"},
      {capacity_bytes = 4 : i64, data_type = "float32", index = 0 : i32, logical_bytes = 4 : i64, name = "x"},
      {capacity_bytes = 4 : i64, data_type = "float32", index = 0 : i32, logical_bytes = 4 : i64, name = "x"}
    ],
    outputs = []
  }
} {
}

// -----

// The plan has at least one buffer per public tensor.

// expected-error @below {{ONNX mutable handle capacity is inconsistent with I/O}}
module attributes {
  udoncl.mutable_buffer_bytes = array<i64: 4>,
  udoncl.mutable_handle_capacity = 1 : i64,
  udoncl.onnx_binding = {
    inputs = [{capacity_bytes = 4 : i64, data_type = "float32", index = 0 : i32, logical_bytes = 4 : i64, name = "x"}],
    outputs = [{capacity_bytes = 4 : i64, data_type = "float32", index = 0 : i32, logical_bytes = 4 : i64, name = "y"}]
  }
} {
}

// -----

// A public tensor's buffer has the planned capacity.

// expected-error @below {{generated ONNX public buffer 1 has inconsistent physical capacity}}
module attributes {
  udoncl.mutable_buffer_bytes = array<i64: 8>,
  udoncl.mutable_handle_capacity = 1 : i64,
  udoncl.onnx_binding = {
    inputs = [{capacity_bytes = 4 : i64, data_type = "float32", index = 0 : i32, logical_bytes = 4 : i64, name = "x"}],
    outputs = []
  }
} {
}

// -----

// Every planned buffer has a positive capacity.

// expected-error @below {{generated ONNX buffer has a non-positive capacity}}
module attributes {
  udoncl.mutable_buffer_bytes = array<i64: 4, 0>,
  udoncl.mutable_handle_capacity = 2 : i64,
  udoncl.onnx_binding = {
    inputs = [{capacity_bytes = 4 : i64, data_type = "float32", index = 0 : i32, logical_bytes = 4 : i64, name = "x"}],
    outputs = []
  }
} {
}

// -----

// A private buffer cannot take the name of a public one.

// expected-error @below {{generated ONNX private buffer name collision}}
module attributes {
  udoncl.mutable_buffer_bytes = array<i64: 4, 4>,
  udoncl.mutable_handle_capacity = 2 : i64,
  udoncl.onnx_binding = {
    inputs = [{capacity_bytes = 4 : i64, data_type = "float32", index = 0 : i32, logical_bytes = 4 : i64, name = "private_2"}],
    outputs = []
  }
} {
}
