// RUN: not %vrc-opt --udoncl-materialize-onnx-host-globals %s 2>&1 | %FileCheck %s

// Every reader of the plan rejects fusion roots that no fused kernel could
// evaluate. Here the product is a graph output, so it must be materialized
// and cannot be an interior node of the Add's group.

module attributes {
  udoncl.onnx_host_graph_validated,
  udoncl.onnx_host_plan = {
    device_fusion_roots = array<i64: 1, 1>,
    device_graph = true,
    device_storage = array<i64: 0, 1, 2>,
    mutable_buffer_bytes = array<i64: 16, 16, 16>,
    mutable_handle_capacity = 3 : i64,
    operator_scratch_bytes = 0 : i64,
    owned_device_buffers = array<i64>
  }
} {
  func.func private @__udon_onnx_mul(
      tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
      attributes {
        udon.onnx.operation = "Mul",
        udon.onnx.provider = "udonatenMul"
      }
  func.func private @__udon_onnx_add(
      tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
      attributes {
        udon.onnx.operation = "Add",
        udon.onnx.provider = "udonatenAdd"
      }
  func.func @main_graph(
      %input: tensor<1x4xf32> {onnx.name = "input"})
      -> (tensor<1x4xf32> {onnx.name = "product"},
          tensor<1x4xf32> {onnx.name = "sum"}) {
    %product = func.call @__udon_onnx_mul(%input, %input)
        : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
    %sum = func.call @__udon_onnx_add(%product, %input)
        : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
    return %product, %sum : tensor<1x4xf32>, tensor<1x4xf32>
  }
}

// CHECK: error: ONNX Host graph plan has invalid fusion roots
