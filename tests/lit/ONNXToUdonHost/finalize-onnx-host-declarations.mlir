// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph --udoncl-materialize-onnx-host-globals --udoncl-materialize-onnx-host-model-registry --udoncl-attach-onnx-host-metadata --udoncl-finalize-onnx-host --udoncl-cleanup-onnx-host-source %s | %FileCheck %s

func.func private @clEnqueueMarkerWithWaitList(
    !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32

func.func private @__udon_onnx_add(
    tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
    attributes {
      udon.onnx.operation = "Add",
      udon.onnx.provider = "udonatenAdd"
    }

func.func @main_graph(
    %left: tensor<1x4xf32> {onnx.name = "left"},
    %right: tensor<1x4xf32> {onnx.name = "right"})
    -> (tensor<1x4xf32> {onnx.name = "sum"}) {
  %sum = func.call @__udon_onnx_add(%left, %right)
      : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
  return %sum : tensor<1x4xf32>
}

// CHECK-COUNT-1: func.func private @clEnqueueMarkerWithWaitList

// Every binding function is a root of the Host but never a U# export.
// CHECK-LABEL: func.func @__udononnx_transitional_create(
// CHECK-SAME: attributes {udoncl.export, udoncl.internal_export}
// CHECK-LABEL: func.func @__udononnx_transitional_bind_all()
// CHECK-SAME: attributes {udoncl.export, udoncl.internal_export}
// CHECK-LABEL: func.func @__udononnx_transitional_enqueue_run()
// CHECK-SAME: attributes {udoncl.export, udoncl.internal_export,
// CHECK-LABEL: func.func @__udononnx_transitional_dispose()
// CHECK-SAME: attributes {udoncl.export, udoncl.internal_export}
