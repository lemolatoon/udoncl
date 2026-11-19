// RUN: not %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph %s 2>&1 | %FileCheck %s

// A Host fallback graph cannot call a Device-pointer provider. The diagnostic
// names the first node that prevents a Device graph.

func.func private @__udon_onnx_gemm(
    tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
    attributes {
      udon.onnx.operation = "Gemm",
      udon.onnx.provider = "udonblasSgemm"
    }

func.func private @__udon_onnx_add(
    tensor<2x4xi32>, tensor<2x4xi32>) -> tensor<2x4xi32>
    attributes {
      udon.onnx.operation = "Add",
      udon.onnx.provider = "udonatenAdd"
    }

func.func @main_graph(
    %left: tensor<2x3xf32> {onnx.name = "left"},
    %right: tensor<3x4xf32> {onnx.name = "right"},
    %a: tensor<2x4xi32> {onnx.name = "a"},
    %b: tensor<2x4xi32> {onnx.name = "b"})
    -> (tensor<2x4xf32> {onnx.name = "product"},
        tensor<2x4xi32> {onnx.name = "sum"}) {
  %product = func.call @__udon_onnx_gemm(%left, %right)
      : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
  %sum = func.call @__udon_onnx_add(%a, %b)
      : (tensor<2x4xi32>, tensor<2x4xi32>) -> tensor<2x4xi32>
  return %product, %sum : tensor<2x4xf32>, tensor<2x4xi32>
}

// CHECK: error: ONNX Host fallback cannot call Device-pointer provider udonblasSgemm for Gemm
// CHECK: note: first Device-unsupported node is Add via udonatenAdd with inputs ('tensor<2x4xi32>', 'tensor<2x4xi32>') and result 'tensor<2x4xi32>'
