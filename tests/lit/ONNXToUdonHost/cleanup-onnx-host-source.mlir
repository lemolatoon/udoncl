// RUN: %vrc-opt --udoncl-cleanup-onnx-host-source %s | %FileCheck %s

module attributes {
  udoncl.onnx_host_finalized,
  udoncl.onnx_host_globals_materialized,
  udoncl.onnx_host_graph_validated,
  udoncl.onnx_host_plan = {}
} {
  func.func private @__udon_onnx_add(tensor<1xf32>) -> tensor<1xf32>
      attributes {
        udon.onnx.operation = "Add",
        udon.onnx.provider = "udonatenAdd"
      }
  func.func private @__udon_onnx_static() -> tensor<1xf32>
      attributes {
        udoncl.static_tensor = {
          byte_length = 4 : i64,
          id = 0 : i64
        }
      }
  func.func @main_graph(%input: tensor<1xf32>) -> tensor<1xf32> {
    %weight = func.call @__udon_onnx_static() : () -> tensor<1xf32>
    %output = func.call @__udon_onnx_add(%input)
        : (tensor<1xf32>) -> tensor<1xf32>
    return %output : tensor<1xf32>
  }
  func.func @other_source_graph(%input: tensor<1xf32>) -> tensor<1xf32>
      attributes {udoncl.onnx_host_source} {
    %output = func.call @__udon_onnx_add(%input)
        : (tensor<1xf32>) -> tensor<1xf32>
    return %output : tensor<1xf32>
  }
  func.func @unrelated(%value: i32) -> i32 {
    return %value : i32
  }
  func.func private @operation_metadata_only()
      attributes {udon.onnx.operation = "Unrelated"}
  func.func private @provider_metadata_only()
      attributes {udon.onnx.provider = "unrelatedProvider"}
  func.func private @runtime_declaration(i32) -> i32
}

// CHECK: module attributes {
// CHECK-SAME: udoncl.onnx_host_finalized
// CHECK-SAME: udoncl.onnx_host_source_cleaned
// CHECK-NOT: udoncl.onnx_host_globals_materialized
// CHECK-NOT: udoncl.onnx_host_graph_validated
// CHECK-NOT: udoncl.onnx_host_plan
// CHECK-NOT: @__udon_onnx_add
// CHECK-NOT: @__udon_onnx_static
// CHECK-NOT: @main_graph
// CHECK-NOT: @other_source_graph
// CHECK: func.func @unrelated
// CHECK: func.func private @operation_metadata_only
// CHECK: func.func private @provider_metadata_only
// CHECK: func.func private @runtime_declaration
