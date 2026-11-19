// RUN: %vrc-opt --udoncl-attach-onnx-host-metadata --verify-diagnostics --split-input-file %s

// expected-error @below {{ONNX Host metadata requires graph validation}}
module {
}

// -----

// expected-error @below {{ONNX Host metadata requires graph planning}}
module attributes {udoncl.onnx_host_graph_validated} {
  func.func @main_graph(%input: tensor<1xf32>) -> tensor<1xf32> {
    return %input : tensor<1xf32>
  }
}

// -----

// expected-error @below {{ONNX Host metadata has already been attached}}
module attributes {udoncl.onnx_host_metadata_attached} {
}
