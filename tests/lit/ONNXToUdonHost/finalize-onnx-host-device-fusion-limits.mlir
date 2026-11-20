// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph %s | %FileCheck %s --check-prefix=PLAN
// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph --udoncl-materialize-onnx-host-globals --udoncl-materialize-onnx-host-model-registry --udoncl-attach-onnx-host-metadata --udoncl-finalize-onnx-host --udoncl-cleanup-onnx-host-source --canonicalize %s | %FileCheck %s
// RUN: %vrc-opt --udoncl-validate-onnx-host-graph --udoncl-plan-onnx-host-graph --udoncl-materialize-onnx-host-globals --udoncl-materialize-onnx-host-model-registry --udoncl-attach-onnx-host-metadata --udoncl-finalize-onnx-host --udoncl-cleanup-onnx-host-source --canonicalize %s | %vrc-opt --vrc-assign-synthesized-kernel-ids=first-id=4096 --vrc-prove-synthesized-gather-kernels | %FileCheck %s --check-prefix=PROVEN

// Where a fusion group stops: a result with a reader outside the group or
// that is a graph output is materialized, a node on its own keeps its
// library kernel, a group that would only absorb metadata aliases is not
// formed, and each cap of the cost model (leaf buffers, Loads on a path,
// expression nodes) splits a tree that exceeds it.

func.func private @clEnqueueMarkerWithWaitList(
    !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32

func.func private @__udon_onnx_mul(tensor<1x4x1x8xf32>, tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
    attributes {
      udon.onnx.operation = "Mul",
      udon.onnx.provider = "udonatenMul"
    }
func.func private @__udon_onnx_add(tensor<1x4x1x8xf32>, tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
    attributes {
      udon.onnx.operation = "Add",
      udon.onnx.provider = "udonatenAdd"
    }
func.func private @__udon_onnx_neg(tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
    attributes {
      udon.onnx.operation = "Neg",
      udon.onnx.provider = "udononnxNeg"
    }
func.func private @__udon_onnx_sin(tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
    attributes {
      udon.onnx.operation = "Sin",
      udon.onnx.provider = "udononnxSin"
    }
func.func private @__udon_onnx_reshape(
    tensor<1x4x1x8xf32>, tensor<2xi64>) -> tensor<4x8xf32>
    attributes {
      udon.onnx.operation = "Reshape",
      udon.onnx.provider = "udononnxReshape"
    }
func.func private @__udon_onnx_neg_rows(tensor<4x8xf32>) -> tensor<4x8xf32>
    attributes {
      udon.onnx.operation = "Neg",
      udon.onnx.provider = "udononnxNeg"
    }
func.func private @__static_rows_shape() -> tensor<2xi64>
    attributes {
      udoncl.static_tensor = {
        byte_length = 16 : i64, id = 0 : i64,
        data = dense<[4, 8]> : tensor<2xi64>
      }
    }

func.func @main_graph(
    %x: tensor<1x4x1x8xf32> {onnx.name = "x"},
    %y: tensor<1x4x1x8xf32> {onnx.name = "y"},
    %v0: tensor<1x4x1x8xf32> {onnx.name = "v0"},
    %v1: tensor<1x4x1x8xf32> {onnx.name = "v1"},
    %v2: tensor<1x4x1x8xf32> {onnx.name = "v2"},
    %v3: tensor<1x4x1x8xf32> {onnx.name = "v3"},
    %v4: tensor<1x4x1x8xf32> {onnx.name = "v4"},
    %v5: tensor<1x4x1x8xf32> {onnx.name = "v5"},
    %v6: tensor<1x4x1x8xf32> {onnx.name = "v6"},
    %v7: tensor<1x4x1x8xf32> {onnx.name = "v7"},
    %v8: tensor<1x4x1x8xf32> {onnx.name = "v8"},
    %v9: tensor<1x4x1x8xf32> {onnx.name = "v9"})
    -> (tensor<1x4x1x8xf32> {onnx.name = "sum"},
        tensor<1x4x1x8xf32> {onnx.name = "negated"},
        tensor<1x4x1x8xf32> {onnx.name = "product"},
        tensor<1x4x1x8xf32> {onnx.name = "product_sum"},
        tensor<4x8xf32> {onnx.name = "rows"},
        tensor<1x4x1x8xf32> {onnx.name = "total"},
        tensor<1x4x1x8xf32> {onnx.name = "power"},
        tensor<1x4x1x8xf32> {onnx.name = "wave"}) {
  // A product read by two groups is materialized, and its readers, each a
  // node on its own, dispatch their library kernels.
  %shared = func.call @__udon_onnx_mul(%x, %y)
      : (tensor<1x4x1x8xf32>, tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %sum = func.call @__udon_onnx_add(%shared, %x)
      : (tensor<1x4x1x8xf32>, tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %negated = func.call @__udon_onnx_neg(%shared)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>

  // A graph output is materialized even if its only reader could fuse it.
  %product = func.call @__udon_onnx_mul(%x, %y)
      : (tensor<1x4x1x8xf32>, tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %product_sum = func.call @__udon_onnx_add(%product, %y)
      : (tensor<1x4x1x8xf32>, tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>

  // Fusing a metadata alias alone would save no dispatch.
  %rows_shape = func.call @__static_rows_shape() : () -> tensor<2xi64>
  %reshaped = func.call @__udon_onnx_reshape(%x, %rows_shape)
      : (tensor<1x4x1x8xf32>, tensor<2xi64>) -> tensor<4x8xf32>
  %rows = func.call @__udon_onnx_neg_rows(%reshaped)
      : (tensor<4x8xf32>) -> tensor<4x8xf32>

  // Ten buffers summed by nine Adds: the root takes the six Adds that keep
  // it at eight leaf buffers, the rest form a second group.
  %total1 = func.call @__udon_onnx_add(%v0, %v1)
      : (tensor<1x4x1x8xf32>, tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %total2 = func.call @__udon_onnx_add(%total1, %v2)
      : (tensor<1x4x1x8xf32>, tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %total3 = func.call @__udon_onnx_add(%total2, %v3)
      : (tensor<1x4x1x8xf32>, tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %total4 = func.call @__udon_onnx_add(%total3, %v4)
      : (tensor<1x4x1x8xf32>, tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %total5 = func.call @__udon_onnx_add(%total4, %v5)
      : (tensor<1x4x1x8xf32>, tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %total6 = func.call @__udon_onnx_add(%total5, %v6)
      : (tensor<1x4x1x8xf32>, tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %total7 = func.call @__udon_onnx_add(%total6, %v7)
      : (tensor<1x4x1x8xf32>, tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %total8 = func.call @__udon_onnx_add(%total7, %v8)
      : (tensor<1x4x1x8xf32>, tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %total = func.call @__udon_onnx_add(%total8, %v9)
      : (tensor<1x4x1x8xf32>, tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>

  // Squaring seven times doubles the Loads of a fragment each time: the
  // root stops at sixteen.
  %power1 = func.call @__udon_onnx_mul(%x, %y)
      : (tensor<1x4x1x8xf32>, tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %power2 = func.call @__udon_onnx_mul(%power1, %power1)
      : (tensor<1x4x1x8xf32>, tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %power3 = func.call @__udon_onnx_mul(%power2, %power2)
      : (tensor<1x4x1x8xf32>, tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %power4 = func.call @__udon_onnx_mul(%power3, %power3)
      : (tensor<1x4x1x8xf32>, tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %power5 = func.call @__udon_onnx_mul(%power4, %power4)
      : (tensor<1x4x1x8xf32>, tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %power6 = func.call @__udon_onnx_mul(%power5, %power5)
      : (tensor<1x4x1x8xf32>, tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %power = func.call @__udon_onnx_mul(%power6, %power6)
      : (tensor<1x4x1x8xf32>, tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>

  // A chain of 66 nodes: 64 in the kernel of the root, two in another.
  %wave1 = func.call @__udon_onnx_sin(%x)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave2 = func.call @__udon_onnx_sin(%wave1)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave3 = func.call @__udon_onnx_sin(%wave2)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave4 = func.call @__udon_onnx_sin(%wave3)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave5 = func.call @__udon_onnx_sin(%wave4)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave6 = func.call @__udon_onnx_sin(%wave5)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave7 = func.call @__udon_onnx_sin(%wave6)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave8 = func.call @__udon_onnx_sin(%wave7)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave9 = func.call @__udon_onnx_sin(%wave8)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave10 = func.call @__udon_onnx_sin(%wave9)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave11 = func.call @__udon_onnx_sin(%wave10)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave12 = func.call @__udon_onnx_sin(%wave11)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave13 = func.call @__udon_onnx_sin(%wave12)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave14 = func.call @__udon_onnx_sin(%wave13)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave15 = func.call @__udon_onnx_sin(%wave14)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave16 = func.call @__udon_onnx_sin(%wave15)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave17 = func.call @__udon_onnx_sin(%wave16)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave18 = func.call @__udon_onnx_sin(%wave17)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave19 = func.call @__udon_onnx_sin(%wave18)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave20 = func.call @__udon_onnx_sin(%wave19)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave21 = func.call @__udon_onnx_sin(%wave20)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave22 = func.call @__udon_onnx_sin(%wave21)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave23 = func.call @__udon_onnx_sin(%wave22)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave24 = func.call @__udon_onnx_sin(%wave23)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave25 = func.call @__udon_onnx_sin(%wave24)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave26 = func.call @__udon_onnx_sin(%wave25)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave27 = func.call @__udon_onnx_sin(%wave26)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave28 = func.call @__udon_onnx_sin(%wave27)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave29 = func.call @__udon_onnx_sin(%wave28)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave30 = func.call @__udon_onnx_sin(%wave29)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave31 = func.call @__udon_onnx_sin(%wave30)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave32 = func.call @__udon_onnx_sin(%wave31)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave33 = func.call @__udon_onnx_sin(%wave32)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave34 = func.call @__udon_onnx_sin(%wave33)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave35 = func.call @__udon_onnx_sin(%wave34)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave36 = func.call @__udon_onnx_sin(%wave35)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave37 = func.call @__udon_onnx_sin(%wave36)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave38 = func.call @__udon_onnx_sin(%wave37)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave39 = func.call @__udon_onnx_sin(%wave38)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave40 = func.call @__udon_onnx_sin(%wave39)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave41 = func.call @__udon_onnx_sin(%wave40)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave42 = func.call @__udon_onnx_sin(%wave41)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave43 = func.call @__udon_onnx_sin(%wave42)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave44 = func.call @__udon_onnx_sin(%wave43)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave45 = func.call @__udon_onnx_sin(%wave44)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave46 = func.call @__udon_onnx_sin(%wave45)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave47 = func.call @__udon_onnx_sin(%wave46)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave48 = func.call @__udon_onnx_sin(%wave47)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave49 = func.call @__udon_onnx_sin(%wave48)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave50 = func.call @__udon_onnx_sin(%wave49)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave51 = func.call @__udon_onnx_sin(%wave50)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave52 = func.call @__udon_onnx_sin(%wave51)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave53 = func.call @__udon_onnx_sin(%wave52)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave54 = func.call @__udon_onnx_sin(%wave53)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave55 = func.call @__udon_onnx_sin(%wave54)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave56 = func.call @__udon_onnx_sin(%wave55)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave57 = func.call @__udon_onnx_sin(%wave56)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave58 = func.call @__udon_onnx_sin(%wave57)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave59 = func.call @__udon_onnx_sin(%wave58)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave60 = func.call @__udon_onnx_sin(%wave59)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave61 = func.call @__udon_onnx_sin(%wave60)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave62 = func.call @__udon_onnx_sin(%wave61)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave63 = func.call @__udon_onnx_sin(%wave62)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave64 = func.call @__udon_onnx_sin(%wave63)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave65 = func.call @__udon_onnx_sin(%wave64)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  %wave = func.call @__udon_onnx_sin(%wave65)
      : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
  return %sum, %negated, %product, %product_sum, %rows, %total, %power, %wave
      : tensor<1x4x1x8xf32>, tensor<1x4x1x8xf32>, tensor<1x4x1x8xf32>, tensor<1x4x1x8xf32>, tensor<4x8xf32>, tensor<1x4x1x8xf32>, tensor<1x4x1x8xf32>, tensor<1x4x1x8xf32>
}

// Nodes in graph order: shared, sum, negated; product, product_sum;
// reshaped, rows; nine Adds; seven Muls; 66 Sins.
// PLAN: device_fusion_roots = array<i64: 0, 1, 2, 3, 4, 5, 6, 8, 8, 15, 15, 15, 15, 15, 15, 15, 18, 18, 18, 22, 22, 22, 22, 24, 24, {{(88, ){63}88>}}

// CHECK-LABEL: func.func @udononnxGeneratedEnqueueRun
// CHECK: call @__vrc_opencl_dispatch_udondnn_large_op_tensor(%[[X:[^,]*]], %[[Y:[^,]*]], %[[X]], %[[SHARED:[^,]*]],
// CHECK: call @__vrc_opencl_dispatch_udondnn_large_op_tensor(%[[SHARED]], %[[X]], %[[SHARED]], %{{[^,]*}},
// CHECK: call @__vrc_opencl_dispatch_udononnx_unary_f32(%[[SHARED]], %{{[^,]*}},

// CHECK: call @__vrc_opencl_dispatch_udondnn_large_op_tensor(%[[X]], %[[Y]], %[[X]], %[[PRODUCT:[^,]*]],
// CHECK: call @__vrc_opencl_dispatch_udondnn_large_op_tensor(%[[PRODUCT]], %[[Y]], %[[PRODUCT]], %{{[^,]*}},

// The Neg reads the buffer of x through the Reshape.
// CHECK: call @__vrc_opencl_dispatch_udononnx_unary_f32(%[[X]], %{{[^,]*}},

// v0 + v1 + v2, then that sum and v3 to v9: eight leaf buffers.
// CHECK: call @__vrc_opencl_dispatch_udononnx_fused_0(%{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %[[TOTAL2:[^,]*]]) : (i32, i32, i32, i32) -> i32
// CHECK: call @__vrc_opencl_dispatch_udononnx_fused_1(%[[TOTAL2]], %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}) : (i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32

// power3 = ((x * y) ^ 2) ^ 2, then power3 ^ 16.
// CHECK: call @__vrc_opencl_dispatch_udononnx_fused_2(%[[X]], %[[Y]], %[[POWER3:[^,]*]]) : (i32, i32, i32) -> i32
// CHECK: call @__vrc_opencl_dispatch_udononnx_fused_3(%[[POWER3]], %{{[^,]*}}) : (i32, i32) -> i32

// wave2 = sin(sin(x)), then 64 more.
// CHECK: call @__vrc_opencl_dispatch_udononnx_fused_4(%[[X]], %[[WAVE2:[^,]*]]) : (i32, i32) -> i32
// CHECK: call @__vrc_opencl_dispatch_udononnx_fused_5(%[[WAVE2]], %{{[^,]*}}) : (i32, i32) -> i32
// CHECK-NOT: call @__vrc_opencl_dispatch_

// The cost model counts a Load per reader; the kernel shares the Load of
// operands that read the same element, so its declared cost can be lower.
// CHECK-LABEL: gpu.module @udononnx_fused_kernels {
// CHECK: gpu.func @udononnx_fused_0(
// CHECK-SAME: vrc.fragment_cost = {loads = 3 : i64}
// CHECK: gpu.func @udononnx_fused_1(
// CHECK-SAME: vrc.fragment_cost = {loads = 8 : i64}
// CHECK: gpu.func @udononnx_fused_2(
// CHECK-SAME: vrc.fragment_cost = {loads = 2 : i64}
// CHECK: gpu.func @udononnx_fused_3(
// CHECK-SAME: vrc.fragment_cost = {loads = 1 : i64}
// CHECK: gpu.func @udononnx_fused_4(
// CHECK-COUNT-2: math.sin
// CHECK: gpu.func @udononnx_fused_5(
// CHECK-COUNT-64: math.sin
// CHECK-NOT: math.sin
// CHECK-NOT: gpu.func

// PROVEN-COUNT-6: vrc.gather_proven
