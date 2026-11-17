// RUN: not %vrc-opt %s "--udoncl-specialize-onnx-inputs=input-shapes=input_ids=1" 2>&1 | %FileCheck %s

module {
  func.func @main_graph(%input_ids: tensor<?x?xi64> {onnx.name = "input_ids"}) {
    return
  }
}

// CHECK: error: ONNX input shape rank mismatch for 'input_ids'
