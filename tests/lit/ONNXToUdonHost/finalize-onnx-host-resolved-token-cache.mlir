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

// No dispatcher enqueues a marker for the creation.
// CHECK-NOT: udononnxGeneratedEnqueueInitialize

// Disposing forgets the buffers a UdonONNX session bound.
// CHECK-LABEL: func.func @udononnxGeneratedDispose
// CHECK: %[[FORGET_INPUT:.*]] = udonhl.address_of @__udon_onnx_bound_input_0
// CHECK-NEXT: arith.constant 0
// CHECK-NEXT: %[[NULL:.*]] = "udonhl.inttoptr"
// CHECK-NEXT: "udonhl.store"(%[[NULL]], %[[FORGET_INPUT]])

// A UdonONNX session keeps the cl_mem it binds to each graph input and
// output in a generated memory cell.
// CHECK-LABEL: func.func @udononnxGeneratedBindMemory(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: !udonhl.ptr<0>) -> i32
// CHECK: cf.switch %arg2 : i32, [
// CHECK: udonhl.address_of @__udon_onnx_bound_input_0
// CHECK-NEXT: "udonhl.store"(%arg3,

// The UdonONNX session run resolves the memory cells (an empty one is a
// missing binding) to tokens, forgets the legacy binding version and runs
// the resolved run.
// CHECK-LABEL: func.func @udononnxGeneratedRun(%arg0: i32, %arg1: !udonhl.ptr<0>, %arg2: !udonhl.ptr<0>) -> i32
// CHECK: %[[CELL:.*]] = udonhl.address_of @__udon_onnx_bound_input_0
// CHECK-NEXT: %[[MEMORY:.*]] = "udonhl.load"(%[[CELL]])
// CHECK: cf.cond_br %{{.*}}, ^[[BOUND:bb[0-9]+]], ^[[MISSING:bb[0-9]+]]
// CHECK: ^[[BOUND]]:
// CHECK: call @udonclResolvePhysicalBuffer(%[[MEMORY]]
// CHECK: ^[[MISSING]]:
// CHECK-NEXT: %[[MISSING_STATUS:.*]] = arith.constant 11 : i32
// CHECK-NEXT: return %[[MISSING_STATUS]]
// CHECK: udonhl.address_of @__udon_onnx_resolved_binding_version
// CHECK: call @__udononnx_enqueue_resolved_run(%arg0, %{{.*}}, %arg1, %{{.*}}, %{{.*}}, %arg2)

// The legacy run resolves the tensor views of its session only when the
// binding version changed.
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
// CHECK-NEXT: %[[RESULT:.*]] = call @__udononnx_enqueue_resolved_run(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5)
// CHECK-NEXT: return %[[RESULT]]
// CHECK: udonhl.address_of @__udon_onnx_resolved_token_2
// CHECK-NEXT: "udonhl.store"
// CHECK-NEXT: %[[STORE_KEY:.*]] = udonhl.address_of @__udon_onnx_resolved_binding_version
// CHECK-NEXT: "udonhl.store"(%[[VERSION]], %[[STORE_KEY]])
// CHECK-NEXT: cf.br ^[[READY]]

// Both runs share the resolved run, which reads the cached tokens.
// CHECK-LABEL: func.func private @__udononnx_enqueue_resolved_run
// CHECK-NEXT: udonhl.address_of @__udon_onnx_resolved_token_0
// CHECK-NEXT: "udonhl.load"
// CHECK: udonhl.address_of @__udon_onnx_resolved_token_2
// CHECK-NEXT: "udonhl.load"

// CHECK: func.func private @udononnxSessionBindingVersion(!udonhl.ptr<0>) -> i32
