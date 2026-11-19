// RUN: %vrc-opt --udoncl-externalize-onnx-graph-constants --udoncl-lower-onnx-to-host --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph --udoncl-materialize-onnx-host-globals --udoncl-materialize-onnx-host-model-registry --udoncl-attach-onnx-host-metadata --udoncl-finalize-onnx-host %s | %FileCheck %s

// A Device dispatch whose operand is a graph-local Constant node reads the
// immutable static tensor. No mutable Device buffer is allocated for the
// constant, so the dispatch never reads storage that nothing has written.

// CHECK-NOT: __udon_onnx_device_buffer_
// CHECK: call @__vrc_opencl_static_tensor
// CHECK-NOT: __udon_onnx_device_buffer_
module attributes {udoncl.static_tensors = []} {
  func.func @main_graph(%x: tensor<1x1x4x4xf32>) -> tensor<1x1x4x4xf32> {
    %mask = onnx.Constant dense<[[[[0.0, -1.0, -1.0, -1.0], [0.0, 0.0, -1.0, -1.0], [0.0, 0.0, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0]]]]> : tensor<1x1x4x4xf32>
    %y = "onnx.Add"(%x, %mask) : (tensor<1x1x4x4xf32>, tensor<1x1x4x4xf32>) -> tensor<1x1x4x4xf32>
    onnx.Return %y : tensor<1x1x4x4xf32>
  }
}
