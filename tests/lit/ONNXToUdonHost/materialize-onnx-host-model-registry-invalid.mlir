// RUN: %vrc-opt --udoncl-materialize-onnx-host-model-registry --verify-diagnostics --split-input-file %s

// expected-error @below {{ONNX Host model registry materialization requires globals}}
module {
}

// -----

// expected-error @below {{ONNX Host model registry materialization requires graph validation}}
module attributes {udoncl.onnx_host_globals_materialized} {
}

// -----

// expected-error @below {{ONNX Host model registry materialization requires graph planning}}
module attributes {
  udoncl.onnx_host_globals_materialized,
  udoncl.onnx_host_graph_validated
} {
  func.func @main_graph(%input: tensor<1xf32>) -> tensor<1xf32> {
    return %input : tensor<1xf32>
  }
}

// -----

// expected-error @below {{ONNX Host model registry has already been materialized}}
module attributes {udoncl.onnx_host_model_registry_materialized} {
}
