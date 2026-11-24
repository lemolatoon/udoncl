// RUN: not %vrc-opt --udoncl-lower-onnx-to-host %s 2>&1 | %FileCheck %s

// A graph output whose derived extent is not the one the model declares.
module attributes {udoncl.dimension_bounds = {past = 3 : i64}} {
  func.func @main_graph(
      %past: tensor<1x?xf32> {onnx.name = "past", udoncl.dim_exprs = [{axis = 1 : i64, offset = 0 : i64, symbol = "past"}]})
      -> (tensor<1x?xf32> {onnx.dim_params = "1:past + 1", onnx.name = "present"}) {
    "onnx.Return"(%past) : (tensor<1x?xf32>) -> ()
  }
}

// CHECK: error: ONNX graph output 'present' axis 1 has extent past but the model declares past + 1
