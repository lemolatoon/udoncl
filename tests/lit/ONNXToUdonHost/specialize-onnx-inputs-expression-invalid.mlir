// RUN: not %vrc-opt %s --split-input-file "--udoncl-specialize-onnx-inputs=dimension-bounds=past=8;sequence=4" 2>&1 | %FileCheck %s

module {
  func.func @main_graph(
      %mask: tensor<?xi64> {onnx.dim_params = "0:past * 2", onnx.name = "attention_mask"}) {
    return
  }
}

// CHECK: error: ONNX dimension expression 'past * 2' of input 'attention_mask' axis 0 is not supported; only 'name' and 'name + N' are

// -----

module {
  func.func @main_graph(
      %mask: tensor<?x?xi64> {onnx.dim_params = "0:sequence,1:past", onnx.name = "attention_mask"}) {
    return
  }
}

// CHECK: error: ONNX input 'attention_mask' has more than one bounded dynamic axis; at most one is supported

// -----

module {
  func.func @main_graph(
      %ids: tensor<?xi64> {onnx.name = "input_ids"}) {
    return
  }
}

// CHECK: error: ONNX input 'input_ids' axis 0 has a dynamic extent without a dimension name; fix it with INPUT_SHAPES
