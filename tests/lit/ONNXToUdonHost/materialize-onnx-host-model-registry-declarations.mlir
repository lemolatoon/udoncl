// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph --udoncl-materialize-onnx-host-globals --udoncl-materialize-onnx-host-model-registry %s | %FileCheck %s

func.func private @udononnxGeneratedGetModelCount() -> i32
func.func private @udononnxGeneratedGetModel(i32, !udonhl.ptr<0>) -> i32

func.func @main_graph(%input: tensor<1xf32> {onnx.name = "input"})
    -> (tensor<1xf32> {onnx.name = "output"}) {
  return %input : tensor<1xf32>
}

// CHECK-COUNT-1: func.func @udononnxGeneratedGetModelCount() -> i32 {
// CHECK-COUNT-1: func.func @udononnxGeneratedGetModel(%{{.*}}: i32, %{{.*}}: !udonhl.ptr<0>) -> i32 {
