// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --split-input-file --verify-diagnostics %s

// expected-error @+1 {{ONNX Host finalization requires one defined single-block main_graph}}
module {
}

// -----

// expected-error @+1 {{ONNX Host tensor requires a nonnegative static ranked shape}}
func.func @main_graph(%input: tensor<?x4xf32>) -> tensor<?x4xf32> {
  return %input : tensor<?x4xf32>
}
