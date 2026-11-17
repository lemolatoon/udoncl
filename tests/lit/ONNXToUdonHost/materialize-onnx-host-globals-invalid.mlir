// RUN: %vrc-opt --udoncl-materialize-onnx-host-globals --verify-diagnostics --split-input-file %s

// expected-error @below {{ONNX Host global materialization requires graph validation}}
module {
}

// -----

// expected-error @below {{ONNX Host global materialization requires graph planning}}
module attributes {udoncl.onnx_host_graph_validated} {
  func.func @main_graph(%input: tensor<1xf32>) -> tensor<1xf32> {
    return %input : tensor<1xf32>
  }
}

// -----

// expected-error @below {{ONNX Host globals have already been materialized}}
module attributes {
  udoncl.onnx_host_globals_materialized,
  udoncl.onnx_host_graph_validated,
  udoncl.onnx_host_plan = {
    device_graph = true,
    device_storage = array<i64: 0>,
    mutable_buffer_bytes = array<i64: 4>,
    mutable_handle_capacity = 1 : i64,
    operator_scratch_bytes = 0 : i64
  }
} {
  func.func @main_graph(%input: tensor<1xf32>) -> tensor<1xf32> {
    return %input : tensor<1xf32>
  }
}
