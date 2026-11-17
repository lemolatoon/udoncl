// RUN: %vrc-opt --udoncl-cleanup-onnx-host-source --verify-diagnostics --split-input-file %s

// expected-error @below {{ONNX Host source cleanup requires finalization}}
module {
}

// -----

// expected-error @below {{ONNX Host source has already been cleaned}}
module attributes {
  udoncl.onnx_host_finalized,
  udoncl.onnx_host_source_cleaned
} {
}
