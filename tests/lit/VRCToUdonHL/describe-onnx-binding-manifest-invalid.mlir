// RUN: %vrc-opt --split-input-file --verify-diagnostics --vrc-describe-onnx-binding-manifest %s

// The metadata names the ABI, the model and its inputs and outputs.

// expected-error @below {{ONNX binding metadata is incomplete}}
module attributes {
  udoncl.onnx_binding = {abi = "udononnx-runtime-v2", model_index = 0 : i32, model_name = "model_0", inputs = []}
} {
}

// -----

// Every graph tensor states its index, name, type, shape and sizes.

// expected-error @below {{ONNX binding tensor metadata is incomplete}}
module attributes {
  udoncl.onnx_binding = {
    abi = "udononnx-runtime-v2", model_index = 0 : i32, model_name = "model_0",
    inputs = [{capacity_bytes = 4 : i64, data_type = "float32", index = 0 : i32, logical_bytes = 4 : i64, name = "x", rank = 1 : i32}],
    outputs = [],
    physical = {}
  }
} {
}

// -----

// Every symbol of a private binding but the sequence-state ones is named.

// expected-error @below {{ONNX binding metadata has no physical bind_all_symbol}}
module attributes {
  udoncl.onnx_binding = {
    abi = "udononnx-runtime-v2", model_index = 0 : i32, model_name = "model_0",
    inputs = [], outputs = [],
    physical = {create_symbol = "c"}
  }
} {
}
