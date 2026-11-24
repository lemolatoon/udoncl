// RUN: %vrc-opt --vrc-derive-onnx-unity-buffers %s | %FileCheck %s

// A module without ONNX binding metadata gets no description, and a
// description of an earlier run does not survive.

// CHECK-LABEL: module attributes
// CHECK-NOT: vrc.onnx_unity_buffers
// CHECK-SAME: vrc.unrelated
module attributes {
  vrc.onnx_unity_buffers = [],
  vrc.unrelated
} {
}
