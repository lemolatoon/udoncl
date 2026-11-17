// RUN: %vrc-opt --canonicalize %s | %FileCheck %s

// CHECK-LABEL: func.func @identity
// CHECK-SAME: (%[[VALUE:.*]]: i32) -> i32
// CHECK: return %[[VALUE]] : i32
func.func @identity(%value: i32) -> i32 {
  return %value : i32
}
