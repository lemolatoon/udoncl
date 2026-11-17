// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph --udoncl-materialize-onnx-host-globals --udoncl-materialize-onnx-host-model-registry --udoncl-finalize-onnx-host --udoncl-cleanup-onnx-host-source %s | %FileCheck %s

func.func private @__udon_onnx_add(
    tensor<1x4xi32>, tensor<1x4xi32>) -> tensor<1x4xi32>
    attributes {
      udon.onnx.operation = "Add",
      udon.onnx.provider = "udonatenAdd"
    }

func.func @main_graph(
    %left: tensor<1x4xi32> {onnx.name = "left"},
    %right: tensor<1x4xi32> {onnx.name = "right"})
    -> (tensor<1x4xi32> {onnx.name = "output"}) {
  %output = func.call @__udon_onnx_add(%left, %right)
      : (tensor<1x4xi32>, tensor<1x4xi32>) -> tensor<1x4xi32>
  return %output : tensor<1x4xi32>
}

// CHECK: module attributes {
// CHECK-SAME: udoncl.onnx_host_finalized
// CHECK-SAME: udoncl.onnx_host_source_cleaned
// CHECK-NOT: @main_graph
// CHECK-NOT: @__udon_onnx_add
// CHECK: func.func @udononnxGeneratedGetModelCount
// CHECK: func.func @udon_onnx_session_write_input_int32
// CHECK: func.func @udon_onnx_session_read_output_int32
// CHECK: func.func @udon_onnx_session_run
