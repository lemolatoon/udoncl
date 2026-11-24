// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph=fuse-device-nodes=false --udoncl-materialize-onnx-host-globals --udoncl-materialize-onnx-host-model-registry --udoncl-attach-onnx-host-metadata --udoncl-finalize-onnx-host --udoncl-cleanup-onnx-host-source %s | %FileCheck %s

// The Device run resolves its bound inputs and outputs and the Device
// buffers the session owns to physical tokens only when the session binding
// version differs from the one the tokens were resolved for. The tokens and
// that version live in globals; every run reads the tokens from them.
// Initialize, which creates the owned buffers, forgets the version.

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
  %first = func.call @__udon_onnx_add(%left, %right)
      : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
  %sum = func.call @__udon_onnx_add(%first, %right)
      : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
  return %sum : tensor<1x4xf32>
}

// Inputs 0 and 1, output 3 and the owned intermediate 2.
// CHECK-DAG: sym_name = "__udon_onnx_resolved_binding_version"}> : () -> ()
// CHECK-DAG: sym_name = "__udon_onnx_resolved_token_0"}> {vrc.resolved_token_cache}
// CHECK-DAG: sym_name = "__udon_onnx_resolved_token_1"}> {vrc.resolved_token_cache}
// CHECK-DAG: sym_name = "__udon_onnx_resolved_token_2"}> {vrc.resolved_token_cache}
// CHECK-DAG: sym_name = "__udon_onnx_resolved_token_3"}> {vrc.resolved_token_cache}

// Creating the session's Device buffers again makes resolved tokens stale.
// CHECK-LABEL: func.func @udononnxGeneratedCreate
// CHECK: %[[FORGET:.*]] = udonhl.address_of @__udon_onnx_resolved_binding_version
// CHECK: "udonhl.store"(%{{.*}}, %[[FORGET]])
// CHECK-NOT: clEnqueueMarkerWithWaitList
// CHECK: return

// The legacy initialization is Create followed by a completion marker.
// CHECK-LABEL: func.func @udononnxGeneratedEnqueueInitialize
// CHECK: %[[CREATED:.*]] = call @udononnxGeneratedCreate(%arg0, %arg1, %arg2)
// CHECK: cf.cond_br %{{.*}}, ^[[MARK:bb[0-9]+]], ^[[FAIL:bb[0-9]+]]
// CHECK: ^[[MARK]]:
// CHECK: call @clEnqueueMarkerWithWaitList(%arg2, %arg3, %arg4, %arg5)
// CHECK: ^[[FAIL]]:
// CHECK: return %[[CREATED]]

// CHECK-LABEL: func.func @udononnxGeneratedEnqueueRun
// CHECK: %[[VERSION:.*]] = call @udononnxSessionBindingVersion(%arg1)
// CHECK: %[[KEY:.*]] = "udonhl.load"
// CHECK: %[[VALID:.*]] = arith.cmpi ne, %[[KEY]]
// CHECK: %[[SAME:.*]] = arith.cmpi eq, %[[VERSION]], %[[KEY]]
// CHECK: %[[REUSE:.*]] = arith.andi %[[SAME]], %[[VALID]]
// CHECK: cf.cond_br %[[REUSE]], ^[[READY:bb[0-9]+]], ^[[RESOLVE:bb[0-9]+]]
// CHECK: ^[[RESOLVE]]:
// CHECK: call @udononnxSessionResolveInput
// CHECK: ^[[READY]]:
// CHECK-NEXT: udonhl.address_of @__udon_onnx_resolved_token_0
// CHECK-NEXT: "udonhl.load"
// CHECK: udonhl.address_of @__udon_onnx_resolved_token_2
// CHECK-NEXT: "udonhl.load"
// CHECK: udonhl.address_of @__udon_onnx_resolved_token_2
// CHECK-NEXT: "udonhl.store"
// CHECK-NEXT: %[[STORE_KEY:.*]] = udonhl.address_of @__udon_onnx_resolved_binding_version
// CHECK-NEXT: "udonhl.store"(%[[VERSION]], %[[STORE_KEY]])
// CHECK-NEXT: cf.br ^[[READY]]

// CHECK: func.func private @udononnxSessionBindingVersion(!udonhl.ptr<0>) -> i32
