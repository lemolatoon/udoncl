// RUN: not %vrc-opt %s "--udoncl-specialize-onnx-inputs=dimensions=batch=1 dimension-bounds=batch=4" 2>&1 | %FileCheck %s --check-prefix=BOTH
// RUN: not %vrc-opt %s "--udoncl-specialize-onnx-inputs=dimensions=batch=1" 2>&1 | %FileCheck %s --check-prefix=UNBOUND
// RUN: not %vrc-opt %s "--udoncl-specialize-onnx-inputs=dimensions=batch=1 dimension-bounds=past=8;other=2" 2>&1 | %FileCheck %s --check-prefix=UNUSED
// RUN: not %vrc-opt %s "--udoncl-specialize-onnx-inputs=dimensions=batch=1 dimension-bounds=past=0" 2>&1 | %FileCheck %s --check-prefix=ZERO

module {
  func.func @main_graph(
      %past: tensor<?x4x?x64xf32> {onnx.dim_params = "0:batch,2:past", onnx.name = "past_key"}) {
    return
  }
}

// BOTH: error: ONNX dimension 'batch' is both fixed and bounded
// UNBOUND: error: ONNX dimension 'past' of input 'past_key' axis 2 is neither fixed (DIMENSIONS) nor bounded (DIMENSION_BOUNDS)
// UNUSED: error: bounded ONNX dimension 'other' is not a dimension of any model input
// ZERO: error: bounded ONNX dimension 'past' needs a positive bound
