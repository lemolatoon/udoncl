// RUN: %vrc-opt --split-input-file --vrc-derive-onnx-unity-buffers %s | %FileCheck %s --check-prefix=PRIVATE
// RUN: %vrc-opt --split-input-file "--vrc-derive-onnx-unity-buffers=public-tensor-buffers=true" %s | %FileCheck %s --check-prefix=PUBLIC

// Every planned buffer of an ONNX Host is a Unity buffer, in id order. Without
// public tensor buffers each is private_<id>. With them, the graph inputs and
// then the outputs are buffers 1 to N, named after their tensor made a C#
// identifier, with the element type, logical size, access and index of the
// ONNX binding. A description of an earlier run is replaced.

// PRIVATE-LABEL: module attributes
// PRIVATE-SAME: vrc.onnx_unity_buffers = [{byte_size = 32 : i64, id = 1 : i64, name = "private_1"}, {byte_size = 16 : i64, id = 2 : i64, name = "private_2"}, {byte_size = 8 : i64, id = 3 : i64, name = "private_3"}, {byte_size = 64 : i64, id = 4 : i64, name = "private_4"}]

// PUBLIC-LABEL: module attributes
// PUBLIC-SAME: vrc.onnx_unity_buffers = [{byte_size = 32 : i64, element_type = "float32", id = 1 : i64, logical_byte_size = 24 : i64, name = "x", public_access = "input", public_index = 0 : i64}, {byte_size = 16 : i64, element_type = "int64", id = 2 : i64, logical_byte_size = 16 : i64, name = "tensor_0_ids", public_access = "input", public_index = 1 : i64}, {byte_size = 8 : i64, element_type = "int32", id = 3 : i64, logical_byte_size = 8 : i64, name = "x_output_0", public_access = "output", public_index = 0 : i64}, {byte_size = 64 : i64, id = 4 : i64, name = "private_4"}]
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
