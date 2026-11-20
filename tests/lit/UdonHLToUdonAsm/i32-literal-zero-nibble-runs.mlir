// RUN: %vrc-opt --vrc-lower-udonhl-to-udonasm %s | %FileCheck %s

// A materialized SystemInt32 literal, such as the minimum SystemInt32 of the
// static async stage, is built from its hexadecimal digits.
// A run of zero digits is one shift by the width of the whole run instead of
// one shift per digit: 0x80000000 is the digit 8 shifted by 28.

// CHECK-DAG: {initial = "8", name = "[[DIGIT:__vrc_constant_[0-9]+]]", type = "SystemInt32"}
// CHECK-DAG: {initial = "28", name = "[[ZEROS:__vrc_constant_[0-9]+]]", type = "SystemInt32"}

// CHECK: udonasm.push "[[DIGIT]]"
// CHECK-NEXT: udonasm.push "[[RESULT:[^"]+]]"
// CHECK-NEXT: udonasm.copy
// CHECK-NEXT: udonasm.push "[[RESULT]]"
// CHECK-NEXT: udonasm.push "[[ZEROS]]"
// CHECK-NEXT: udonasm.push "[[RESULT]]"
// CHECK-NEXT: udonasm.extern "SystemInt32.__op_LeftShift__SystemInt32_SystemInt32__SystemInt32"
// CHECK-NEXT: udonasm.push "[[RESULT]]"
// CHECK-NEXT: udonasm.push "__0_minimum__ret"
module attributes {vrc.async_stage = "udon-static-v1"} {
  func.func @minimum() -> i32 attributes {vrc.export_id = 1 : i32} {
    %value = arith.constant -2147483648 : i32
    return %value : i32
  }
}
