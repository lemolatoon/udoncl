// RUN: %vrc-opt --udoncl-simplify-onnx-views %s | %FileCheck %s

// A Transpose that only moves unit dimensions becomes a Reshape; equal result
// shapes share one shape constant. A Transpose that reorders two non-unit
// dimensions is kept.

module {
  func.func @transposes(
      %heads: tensor<1x1x32x64xf32>,
      %more_heads: tensor<1x1x32x64xf32>,
      %matrix: tensor<1x32x64x64xf32>)
      -> (tensor<1x32x1x64xf32>, tensor<1x32x1x64xf32>,
          tensor<1x32x64x64xf32>) {
    %moved = "onnx.Transpose"(%heads) {perm = [0, 2, 1, 3]}
        : (tensor<1x1x32x64xf32>) -> tensor<1x32x1x64xf32>
    %moved_again = "onnx.Transpose"(%more_heads) {perm = [0, 2, 1, 3]}
        : (tensor<1x1x32x64xf32>) -> tensor<1x32x1x64xf32>
    %swapped = "onnx.Transpose"(%matrix) {perm = [0, 1, 3, 2]}
        : (tensor<1x32x64x64xf32>) -> tensor<1x32x64x64xf32>
    "onnx.Return"(%moved, %moved_again, %swapped)
        : (tensor<1x32x1x64xf32>, tensor<1x32x1x64xf32>,
           tensor<1x32x64x64xf32>) -> ()
  }

// CHECK-LABEL: func.func @transposes
// CHECK-SAME: (%[[HEADS:.*]]: tensor<1x1x32x64xf32>, %[[MORE:.*]]: tensor<1x1x32x64xf32>, %[[MATRIX:.*]]: tensor<1x32x64x64xf32>)
// CHECK: %[[SHAPE:.*]] = arith.constant dense<[1, 32, 1, 64]> : tensor<4xi64>
// CHECK-NOT: arith.constant
// CHECK: %[[MOVED:.*]] = "onnx.Reshape"(%[[HEADS]], %[[SHAPE]])
// CHECK-SAME: (tensor<1x1x32x64xf32>, tensor<4xi64>) -> tensor<1x32x1x64xf32>
// CHECK: %[[AGAIN:.*]] = "onnx.Reshape"(%[[MORE]], %[[SHAPE]])
// CHECK: %[[SWAPPED:.*]] = "onnx.Transpose"(%[[MATRIX]])
// CHECK: onnx.Return %[[MOVED]], %[[AGAIN]], %[[SWAPPED]]

  // The default permutation reverses the axes; it is a Reshape only while at
  // most one dimension is not a unit dimension.
  func.func @default_permutation(
      %column: tensor<1x64x1xf32>, %plane: tensor<2x64x1xf32>)
      -> (tensor<1x64x1xf32>, tensor<1x64x2xf32>) {
    %row = "onnx.Transpose"(%column)
        : (tensor<1x64x1xf32>) -> tensor<1x64x1xf32>
    %reversed = "onnx.Transpose"(%plane)
        : (tensor<2x64x1xf32>) -> tensor<1x64x2xf32>
    "onnx.Return"(%row, %reversed)
        : (tensor<1x64x1xf32>, tensor<1x64x2xf32>) -> ()
  }

// CHECK-LABEL: func.func @default_permutation
// CHECK-SAME: (%[[COLUMN:.*]]: tensor<1x64x1xf32>, %[[PLANE:.*]]: tensor<2x64x1xf32>)
// CHECK: %[[ROW:.*]] = "onnx.Reshape"(%[[COLUMN]], %{{.*}})
// CHECK: %[[REVERSED:.*]] = "onnx.Transpose"(%[[PLANE]])
// CHECK: onnx.Return %[[ROW]], %[[REVERSED]]

  // A unit-step Slice or an Expand that keeps the input shape is its input.
  // A reversing Slice, a Slice with unknown steps and a narrowing Slice stay.
  func.func @identities(
      %mask: tensor<1x1x1x64xf32>, %steps: tensor<1xi64>)
      -> (tensor<1x1x1x64xf32>, tensor<1x1x1x64xf32>, tensor<1x1x1x64xf32>,
          tensor<1x1x1x64xf32>, tensor<1x1x1x32xf32>, tensor<1x1x1x64xf32>) {
    %none = "onnx.NoValue"() {value} : () -> none
    %zero = "onnx.Constant"() {value = dense<0> : tensor<1xi64>}
        : () -> tensor<1xi64>
    %last = "onnx.Constant"() {value = dense<-1> : tensor<1xi64>}
        : () -> tensor<1xi64>
    %half = "onnx.Constant"() {value = dense<32> : tensor<1xi64>}
        : () -> tensor<1xi64>
    %end = "onnx.Constant"() {value = dense<64> : tensor<1xi64>}
        : () -> tensor<1xi64>
    %before = "onnx.Constant"()
        {value = dense<-9223372036854775808> : tensor<1xi64>}
        : () -> tensor<1xi64>
    %axis = "onnx.Constant"() {value = dense<3> : tensor<1xi64>}
        : () -> tensor<1xi64>
    %one = "onnx.Constant"() {value = dense<1> : tensor<1xi64>}
        : () -> tensor<1xi64>
    %minus_one = "onnx.Constant"() {value = dense<-1> : tensor<1xi64>}
        : () -> tensor<1xi64>
    %shape = "onnx.Constant"() {value = dense<[1, 1, 1, 64]> : tensor<4xi64>}
        : () -> tensor<4xi64>
    %whole = "onnx.Slice"(%mask, %zero, %end, %axis, %one)
        : (tensor<1x1x1x64xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>,
           tensor<1xi64>) -> tensor<1x1x1x64xf32>
    %default_steps = "onnx.Slice"(%mask, %zero, %end, %axis, %none)
        : (tensor<1x1x1x64xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>,
           none) -> tensor<1x1x1x64xf32>
    %reversed = "onnx.Slice"(%mask, %last, %before, %axis, %minus_one)
        : (tensor<1x1x1x64xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>,
           tensor<1xi64>) -> tensor<1x1x1x64xf32>
    %unknown = "onnx.Slice"(%mask, %zero, %end, %axis, %steps)
        : (tensor<1x1x1x64xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>,
           tensor<1xi64>) -> tensor<1x1x1x64xf32>
    %narrow = "onnx.Slice"(%mask, %zero, %half, %axis, %one)
        : (tensor<1x1x1x64xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>,
           tensor<1xi64>) -> tensor<1x1x1x32xf32>
    %expanded = "onnx.Expand"(%mask, %shape)
        : (tensor<1x1x1x64xf32>, tensor<4xi64>) -> tensor<1x1x1x64xf32>
    "onnx.Return"(%whole, %default_steps, %reversed, %unknown, %narrow,
                  %expanded)
        : (tensor<1x1x1x64xf32>, tensor<1x1x1x64xf32>, tensor<1x1x1x64xf32>,
           tensor<1x1x1x64xf32>, tensor<1x1x1x32xf32>, tensor<1x1x1x64xf32>)
        -> ()
  }

// CHECK-LABEL: func.func @identities
// CHECK-SAME: (%[[MASK:.*]]: tensor<1x1x1x64xf32>, %[[STEPS:.*]]: tensor<1xi64>)
// CHECK-NOT: onnx.Expand
// CHECK: %[[REVERSED:.*]] = "onnx.Slice"(%[[MASK]],
// CHECK: %[[UNKNOWN:.*]] = "onnx.Slice"(%[[MASK]],
// CHECK-SAME: %[[STEPS]])
// CHECK: %[[NARROW:.*]] = "onnx.Slice"(%[[MASK]],
// CHECK-NOT: onnx.Expand
// CHECK: onnx.Return %[[MASK]], %[[MASK]], %[[REVERSED]], %[[UNKNOWN]], %[[NARROW]], %[[MASK]]

  // A broadcasting Expand is kept.
  func.func @broadcast(%mask: tensor<1x1x1x64xf32>) -> tensor<1x4x1x64xf32> {
    %shape = "onnx.Constant"() {value = dense<[1, 4, 1, 64]> : tensor<4xi64>}
        : () -> tensor<4xi64>
    %expanded = "onnx.Expand"(%mask, %shape)
        : (tensor<1x1x1x64xf32>, tensor<4xi64>) -> tensor<1x4x1x64xf32>
    "onnx.Return"(%expanded) : (tensor<1x4x1x64xf32>) -> ()
  }

// CHECK-LABEL: func.func @broadcast
// CHECK: "onnx.Expand"

  // A ScatterND whose constant indices overwrite every element in row-major
  // order (negative indices count from the end) is its updates reshaped, with
  // or without the default reduction spelled out. One that reorders an
  // element, or that reduces, is kept.
  func.func @scatters(%data: tensor<1x1x2x2xf32>, %updates: tensor<1x1x2x2xf32>)
      -> (tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32>,
          tensor<1x1x2x2xf32>) {
    %every = "onnx.Constant"() {value = dense<[[[[[0, 0, 0, 0], [0, 0, 0, 1]], [[0, 0, 1, 0], [0, 0, -1, -1]]]]]> : tensor<1x1x2x2x4xi64>}
        : () -> tensor<1x1x2x2x4xi64>
    %swapped = "onnx.Constant"() {value = dense<[[[[[0, 0, 0, 1], [0, 0, 0, 0]], [[0, 0, 1, 0], [0, 0, 1, 1]]]]]> : tensor<1x1x2x2x4xi64>}
        : () -> tensor<1x1x2x2x4xi64>
    %overwritten = "onnx.ScatterND"(%data, %every, %updates)
        {reduction = "none", onnx_node_name = "whole"}
        : (tensor<1x1x2x2xf32>, tensor<1x1x2x2x4xi64>, tensor<1x1x2x2xf32>)
        -> tensor<1x1x2x2xf32>
    %reordered = "onnx.ScatterND"(%data, %swapped, %updates)
        {reduction = "none"}
        : (tensor<1x1x2x2xf32>, tensor<1x1x2x2x4xi64>, tensor<1x1x2x2xf32>)
        -> tensor<1x1x2x2xf32>
    %added = "onnx.ScatterND"(%data, %every, %updates) {reduction = "add"}
        : (tensor<1x1x2x2xf32>, tensor<1x1x2x2x4xi64>, tensor<1x1x2x2xf32>)
        -> tensor<1x1x2x2xf32>
    %unreduced = "onnx.ScatterND"(%data, %every, %updates)
        : (tensor<1x1x2x2xf32>, tensor<1x1x2x2x4xi64>, tensor<1x1x2x2xf32>)
        -> tensor<1x1x2x2xf32>
    "onnx.Return"(%overwritten, %reordered, %added, %unreduced)
        : (tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32>,
           tensor<1x1x2x2xf32>) -> ()
  }

// CHECK-LABEL: func.func @scatters
// CHECK-SAME: (%[[DATA:.*]]: tensor<1x1x2x2xf32>, %[[UPDATES:.*]]: tensor<1x1x2x2xf32>)
// CHECK: %[[WHOLE:.*]] = "onnx.Reshape"(%[[UPDATES]], %{{.*}}){{.*}}onnx_node_name = "whole"
// CHECK: %[[REORDERED:.*]] = "onnx.ScatterND"(%[[DATA]]
// CHECK: %[[ADDED:.*]] = "onnx.ScatterND"(%[[DATA]]
// CHECK: %[[UNREDUCED:.*]] = "onnx.Reshape"(%[[UPDATES]]
// CHECK: onnx.Return %[[WHOLE]], %[[REORDERED]], %[[ADDED]], %[[UNREDUCED]]
}
