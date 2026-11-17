// RUN: %vrc-opt %s "--udoncl-specialize-onnx-inputs=dimensions=batch=1;sequence=4 input-shapes=attention_mask=1,5" | %FileCheck %s

module {
  func.func @main_graph(
      %input_ids: tensor<?x?xi64> {onnx.dim_params = "0:batch,1:sequence", onnx.name = "input_ids"},
      %attention_mask: tensor<?x?xi64> {onnx.dim_params = "0:batch,1:sequence", onnx.name = "attention_mask"}) {
    return
  }
}

// CHECK: func.func @main_graph(%arg0: tensor<1x4xi64> {{.*}}onnx.name = "input_ids"{{.*}}, %arg1: tensor<1x5xi64> {{.*}}onnx.name = "attention_mask"{{.*}})
