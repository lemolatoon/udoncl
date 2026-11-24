// RUN: %vrc-opt %s "--udoncl-specialize-onnx-inputs=dimensions=batch=1 dimension-bounds=past=511" | %FileCheck %s

// A bounded dimension keeps its axis dynamic and records the extent
// expression of that axis; fixed dimensions are specialized as before.

module {
  func.func @main_graph(
      %mask: tensor<?x?xi64> {onnx.dim_params = "0:batch,1:past + 1", onnx.name = "attention_mask"},
      %past: tensor<?x4x?x64xf32> {onnx.dim_params = "0:batch,2:past", onnx.name = "past_key"}) {
    return
  }
}

// CHECK: module attributes {udoncl.dimension_bounds = {past = 511 : i64}}
// CHECK: func.func @main_graph(%arg0: tensor<1x?xi64> {{{.*}}udoncl.dim_exprs = [{axis = 1 : i64, offset = 1 : i64, symbol = "past"}]}, %arg1: tensor<1x4x?x64xf32> {{{.*}}udoncl.dim_exprs = [{axis = 2 : i64, offset = 0 : i64, symbol = "past"}]})
