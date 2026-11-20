// RUN: %vrc-opt --udoncl-merge-onnx-sibling-matmulnbits --mlir-print-debuginfo %s | %FileCheck %s
// RUN: %vrc-opt --udoncl-merge-onnx-sibling-matmulnbits=max-result-words=7 %s | %FileCheck %s --check-prefix=CAP

// Three MatMulNBits nodes read one activation with equal K, bits and block
// size: they become one node over the stacked weight and scale rows, and each
// original result is a Slice of its last axis. Slice operands are shared and
// the separate weights are erased.

module {
  func.func @siblings(%x: tensor<2x32xf32>)
      -> (tensor<2x2xf32>, tensor<2x3xf32>, tensor<2x1xf32>) {
    %none = "onnx.NoValue"() {value} : () -> none
    %bq = onnx.Constant dense<1> : tensor<2x1x16xi8> loc(#q_weights)
    %sq = onnx.Constant dense<[1.0, 2.0]> : tensor<2xf32> loc(#q_scales)
    %bk = onnx.Constant dense<2> : tensor<3x1x16xi8> loc(#k_weights)
    %sk = onnx.Constant dense<[3.0, 4.0, 5.0]> : tensor<3xf32> loc(#k_scales)
    %bv = onnx.Constant dense<3> : tensor<1x1x16xi8> loc(#v_weights)
    %sv = onnx.Constant dense<[6.0]> : tensor<1xf32> loc(#v_scales)
    %q = "onnx.Custom"(%x, %bq, %sq) <{function_name = "MatMulNBits"}>
        {K = 32 : si64, N = 2 : si64, bits = 4 : si64, block_size = 32 : si64,
         domain_name = "com.microsoft", onnx_node_name = "q"}
        : (tensor<2x32xf32>, tensor<2x1x16xi8>, tensor<2xf32>) -> tensor<2x2xf32>
    %k = "onnx.Custom"(%x, %bk, %sk, %none) <{function_name = "MatMulNBits"}>
        {K = 32 : si64, N = 3 : si64, bits = 4 : si64, block_size = 32 : si64,
         domain_name = "com.microsoft", onnx_node_name = "k"}
        : (tensor<2x32xf32>, tensor<3x1x16xi8>, tensor<3xf32>, none) -> tensor<2x3xf32>
    %v = "onnx.Custom"(%x, %bv, %sv) <{function_name = "MatMulNBits"}>
        {K = 32 : si64, N = 1 : si64, bits = 4 : si64, block_size = 32 : si64,
         domain_name = "com.microsoft", onnx_node_name = "v"}
        : (tensor<2x32xf32>, tensor<1x1x16xi8>, tensor<1xf32>) -> tensor<2x1xf32>
    "onnx.Return"(%q, %k, %v) : (tensor<2x2xf32>, tensor<2x3xf32>, tensor<2x1xf32>) -> ()
  }

  // Nothing merges with a node that has a different K, a bias, weights that
  // are not constants, or another activation.
  func.func @kept(%x: tensor<1x64xf32>, %y: tensor<1x64xf32>,
                  %weights: tensor<1x2x16xi8>)
      -> (tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>,
          tensor<1x1xf32>) {
    %b = onnx.Constant dense<1> : tensor<1x2x16xi8>
    %s = onnx.Constant dense<[1.0, 2.0]> : tensor<2xf32>
    %b32 = onnx.Constant dense<1> : tensor<1x1x16xi8>
    %s32 = onnx.Constant dense<[1.0]> : tensor<1xf32>
    %bias = onnx.Constant dense<[0.5]> : tensor<1xf32>
    %none = "onnx.NoValue"() {value} : () -> none
    %plain = "onnx.Custom"(%x, %b, %s) <{function_name = "MatMulNBits"}>
        {K = 64 : si64, N = 1 : si64, bits = 4 : si64, block_size = 32 : si64,
         domain_name = "com.microsoft", onnx_node_name = "plain"}
        : (tensor<1x64xf32>, tensor<1x2x16xi8>, tensor<2xf32>) -> tensor<1x1xf32>
    %wide = "onnx.Custom"(%x, %b32, %s32) <{function_name = "MatMulNBits"}>
        {K = 64 : si64, N = 1 : si64, bits = 4 : si64, block_size = 64 : si64,
         domain_name = "com.microsoft", onnx_node_name = "wide"}
        : (tensor<1x64xf32>, tensor<1x1x16xi8>, tensor<1xf32>) -> tensor<1x1xf32>
    %biased = "onnx.Custom"(%x, %b, %s, %none, %none, %bias)
        <{function_name = "MatMulNBits"}>
        {K = 64 : si64, N = 1 : si64, bits = 4 : si64, block_size = 32 : si64,
         domain_name = "com.microsoft", onnx_node_name = "biased"}
        : (tensor<1x64xf32>, tensor<1x2x16xi8>, tensor<2xf32>, none, none,
           tensor<1xf32>) -> tensor<1x1xf32>
    %dynamic = "onnx.Custom"(%x, %weights, %s) <{function_name = "MatMulNBits"}>
        {K = 64 : si64, N = 1 : si64, bits = 4 : si64, block_size = 32 : si64,
         domain_name = "com.microsoft", onnx_node_name = "dynamic"}
        : (tensor<1x64xf32>, tensor<1x2x16xi8>, tensor<2xf32>) -> tensor<1x1xf32>
    %other = "onnx.Custom"(%y, %b, %s) <{function_name = "MatMulNBits"}>
        {K = 64 : si64, N = 1 : si64, bits = 4 : si64, block_size = 32 : si64,
         domain_name = "com.microsoft", onnx_node_name = "other"}
        : (tensor<1x64xf32>, tensor<1x2x16xi8>, tensor<2xf32>) -> tensor<1x1xf32>
    "onnx.Return"(%plain, %wide, %biased, %dynamic, %other)
        : (tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>,
           tensor<1x1xf32>) -> ()
  }
}

#q_weights = loc("Initializer_q.weight")
#q_scales = loc("Initializer_q.scales")
#k_weights = loc("Initializer_k.weight")
#k_scales = loc("Initializer_k.scales")
#v_weights = loc("Initializer_v.weight")
#v_scales = loc("Initializer_v.scales")

// CHECK-LABEL: func.func @siblings
// CHECK-SAME: (%[[X:[^:]*]]: tensor<2x32xf32>
// CHECK-DAG: %[[ZERO:.*]] = arith.constant dense<0> : tensor<1xi64>
// CHECK-DAG: %[[ONE:.*]] = arith.constant dense<1> : tensor<1xi64>
// CHECK-DAG: %[[TWO:.*]] = arith.constant dense<2> : tensor<1xi64>
// CHECK-DAG: %[[FIVE:.*]] = arith.constant dense<5> : tensor<1xi64>
// CHECK-DAG: %[[SIX:.*]] = arith.constant dense<6> : tensor<1xi64>
// CHECK-NOT: onnx.Constant dense<1> : tensor<2x1x16xi8>
// CHECK: %[[B:.*]] = onnx.Constant dense<{{.*}}> : tensor<6x1x16xi8>
// CHECK: %[[S:.*]] = onnx.Constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf32>
// CHECK: %[[QKV:.*]] = "onnx.Custom"(%[[X]], %[[B]], %[[S]])
// CHECK-SAME: N = 6 : si64
// CHECK-SAME: onnx_node_name = "q+k+v"
// CHECK-SAME: -> tensor<2x6xf32>
// CHECK: %[[Q:.*]] = "onnx.Slice"(%[[QKV]], %[[ZERO]], %[[TWO]], %[[ONE]], %[[ONE]]) {onnx_node_name = "q"}
// CHECK-SAME: -> tensor<2x2xf32>
// CHECK: %[[K:.*]] = "onnx.Slice"(%[[QKV]], %[[TWO]], %[[FIVE]], %[[ONE]], %[[ONE]]) {onnx_node_name = "k"}
// CHECK-SAME: -> tensor<2x3xf32>
// CHECK: %[[V:.*]] = "onnx.Slice"(%[[QKV]], %[[FIVE]], %[[SIX]], %[[ONE]], %[[ONE]]) {onnx_node_name = "v"}
// CHECK-SAME: -> tensor<2x1xf32>
// CHECK-NOT: "onnx.Custom"
// CHECK: onnx.Return %[[Q]], %[[K]], %[[V]]

// CHECK-LABEL: func.func @kept
// CHECK-NOT: onnx.Slice
// CHECK-COUNT-5: "onnx.Custom"
// CHECK-NOT: onnx.Slice

// The stacked rows keep the initializer origin, so the merged weights are
// stored outside the IR like the weights they replace.
// CHECK: loc("Initializer_q.weight+k.weight+v.weight"
// CHECK: loc("Initializer_q.scales+k.scales+v.scales"

// A group stays within max-result-words: 2 rows of q and k are 10 words, so
// with a cap of 7 only q (4) and v (2) merge and k stays separate.
// CAP-LABEL: func.func @siblings
// CAP: "onnx.Custom"({{.*}}) <{function_name = "MatMulNBits"}> {{{.*}}N = 3 : si64{{.*}}onnx_node_name = "q+v"}
// CAP: "onnx.Custom"({{.*}}) <{function_name = "MatMulNBits"}> {{{.*}}N = 3 : si64{{.*}}onnx_node_name = "k"}
