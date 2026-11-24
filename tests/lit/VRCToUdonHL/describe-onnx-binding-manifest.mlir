// RUN: %vrc-opt --split-input-file --vrc-describe-onnx-binding-manifest %s | %FileCheck %s

// The description names the ABI, the model and the graph inputs and outputs.
// A description of an earlier run is replaced.

// CHECK-LABEL: module attributes
// CHECK-SAME: vrc.onnx_binding_manifest = {abi = "udononnx-runtime-v3", inputs = [{capacity_bytes = 64 : i64, data_type = "float32", dimensions = [1, 2, 0, 8], index = 0 : i64, logical_bytes = 0 : i64, name = "past", rank = 4 : i64}], model_index = 0 : i64, model_name = "model_0", outputs = [{capacity_bytes = 64 : i64, data_type = "float32", dimensions = [1, 2, 1, 8], index = 0 : i64, logical_bytes = 64 : i64, name = "present", rank = 4 : i64}]}
module attributes {
  udoncl.onnx_binding = {
    abi = "udononnx-runtime-v3", model_index = 0 : i32, model_name = "model_0",
    inputs = [{capacity_bytes = 64 : i64, data_type = "float32", dimensions = array<i64: 1, 2, 0, 8>, index = 0 : i32, logical_bytes = 0 : i64, name = "past", rank = 4 : i32}],
    outputs = [{capacity_bytes = 64 : i64, data_type = "float32", dimensions = array<i64: 1, 2, 1, 8>, index = 0 : i32, logical_bytes = 64 : i64, name = "present", rank = 4 : i32}]
  },
  vrc.onnx_binding_manifest = {}
} {
}

// -----

// A graph without inputs and outputs.

// CHECK-LABEL: module attributes
// CHECK-SAME: vrc.onnx_binding_manifest = {abi = "udononnx-runtime-v3", inputs = [], model_index = 0 : i64, model_name = "model_0", outputs = []}
module attributes {
  udoncl.onnx_binding = {
    abi = "udononnx-runtime-v3", model_index = 0 : i32, model_name = "model_0",
    inputs = [], outputs = []
  }
} {
}

// -----

// A module without ONNX binding metadata gets no description.

// CHECK-LABEL: module attributes
// CHECK-NOT: vrc.onnx_binding_manifest
// CHECK-SAME: vrc.unrelated
module attributes {
  vrc.onnx_binding_manifest = {},
  vrc.unrelated
} {
}
