// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph --udoncl-materialize-onnx-host-globals --udoncl-materialize-onnx-host-model-registry --verify-diagnostics --split-input-file %s

// expected-error @below {{ONNX Host generated function udononnxGeneratedGetModelCount has type '() -> i64'; expected '() -> i32'}}
func.func private @udononnxGeneratedGetModelCount() -> i64
func.func @main_graph(%input: tensor<1xf32> {onnx.name = "input"})
    -> (tensor<1xf32> {onnx.name = "output"}) {
  return %input : tensor<1xf32>
}

// -----

// expected-error @below {{ONNX Host generated function udononnxGeneratedGetModelCount declaration must have private visibility}}
func.func nested @udononnxGeneratedGetModelCount() -> i32
func.func @main_graph(%input: tensor<1xf32> {onnx.name = "input"})
    -> (tensor<1xf32> {onnx.name = "output"}) {
  return %input : tensor<1xf32>
}

// -----

// expected-error @below {{ONNX Host generated function udononnxGeneratedGetModelCount conflicts with a definition}}
func.func @udononnxGeneratedGetModelCount() -> i32 {
  %one = arith.constant 1 : i32
  return %one : i32
}
func.func @main_graph(%input: tensor<1xf32> {onnx.name = "input"})
    -> (tensor<1xf32> {onnx.name = "output"}) {
  return %input : tensor<1xf32>
}
