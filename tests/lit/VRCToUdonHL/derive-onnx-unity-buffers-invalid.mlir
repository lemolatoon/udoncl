// RUN: %vrc-opt --split-input-file --verify-diagnostics --vrc-derive-onnx-unity-buffers %s

// The plan has one capacity per buffer.

// expected-error @below {{ONNX mutable handle capacity is inconsistent with its buffers}}
module attributes {
  udoncl.mutable_buffer_bytes = array<i64: 4>,
  udoncl.mutable_handle_capacity = 2 : i64,
  udoncl.onnx_binding = {inputs = [], outputs = []}
} {
}

// -----

// Every planned buffer has a positive capacity.

// expected-error @below {{generated ONNX buffer has a non-positive capacity}}
module attributes {
  udoncl.mutable_buffer_bytes = array<i64: 4, 0>,
  udoncl.mutable_handle_capacity = 2 : i64,
  udoncl.onnx_binding = {inputs = [], outputs = []}
} {
}
