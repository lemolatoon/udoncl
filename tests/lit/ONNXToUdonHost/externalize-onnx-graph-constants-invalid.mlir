// RUN: %vrc-opt --udoncl-externalize-onnx-graph-constants --verify-diagnostics %s

// Graph constants share the static tensor id space with initializers, so the
// initializer records must exist first.

// expected-error @below {{graph constant externalization requires ONNX initializer externalization}}
module {
  func.func @main_graph() -> tensor<2xf32> {
    %value = arith.constant dense<[1.0, 2.0]> : tensor<2xf32>
    return %value : tensor<2xf32>
  }
}
