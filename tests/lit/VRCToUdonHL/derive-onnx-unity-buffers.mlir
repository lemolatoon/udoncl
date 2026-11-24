// RUN: %vrc-opt --split-input-file --vrc-derive-onnx-unity-buffers %s | %FileCheck %s

// Every planned buffer of an ONNX Host is a private Unity buffer
// private_<id>, in id order; a Host exposes a graph tensor by binding a
// UDONCL_PUBLIC buffer of its own to it. A description of an earlier run is
// replaced.

// CHECK-LABEL: module attributes
// CHECK-SAME: vrc.onnx_unity_buffers = [{byte_size = 32 : i64, id = 1 : i64, name = "private_1"}, {byte_size = 16 : i64, id = 2 : i64, name = "private_2"}, {byte_size = 8 : i64, id = 3 : i64, name = "private_3"}, {byte_size = 64 : i64, id = 4 : i64, name = "private_4"}]
module attributes {
  vrc.onnx_unity_buffers = [],
  udoncl.mutable_buffer_bytes = array<i64: 32, 16, 8, 64>,
  udoncl.mutable_handle_capacity = 4 : i64,
  udoncl.onnx_binding = {
    inputs = [
      {capacity_bytes = 32 : i64, data_type = "float32", index = 0 : i32, logical_bytes = 24 : i64, name = "x"},
      {capacity_bytes = 16 : i64, data_type = "int64", index = 1 : i32, logical_bytes = 16 : i64, name = "0.ids"}
    ],
    outputs = [
      {capacity_bytes = 8 : i64, data_type = "int32", index = 0 : i32, logical_bytes = 8 : i64, name = "x"}
    ]
  }
} {
}

// -----

// A module without ONNX binding metadata gets no description.

// CHECK-LABEL: module {
// CHECK-NOT: vrc.onnx_unity_buffers
module {
}
