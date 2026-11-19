// RUN: not %vrc-opt --udoncl-lower-onnx-to-host --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph %s 2>&1 | %FileCheck %s

// A Device graph operand must be a graph input, an immutable static tensor,
// or the result of an earlier node. A dense constant that was never
// externalized would be backed by a mutable Device buffer that nothing
// writes, so planning rejects it instead of reading zeros at run time.

// CHECK: error: ONNX Device graph node Add via {{[A-Za-z]+}} reads __udon_onnx_constant_{{[0-9]+}} ('tensor<1x1x4x4xf32>') that is neither a graph input, an immutable static tensor, nor the result of an earlier node
module attributes {udoncl.static_tensors = []} {
  func.func @main_graph(%x: tensor<1x1x4x4xf32>) -> tensor<1x1x4x4xf32> {
    %mask = onnx.Constant dense<[[[[0.0, -1.0, -1.0, -1.0], [0.0, 0.0, -1.0, -1.0], [0.0, 0.0, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0]]]]> : tensor<1x1x4x4xf32>
    %y = "onnx.Add"(%x, %mask) : (tensor<1x1x4x4xf32>, tensor<1x1x4x4xf32>) -> tensor<1x1x4x4xf32>
    onnx.Return %y : tensor<1x1x4x4xf32>
  }
}
