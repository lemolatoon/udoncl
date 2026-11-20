// RUN: %vrc-opt --vrc-lower-udonhl-to-udonasm %s | %FileCheck %s

// arith.trunci keeps the low word of any i64, but SystemConvert.ToInt32
// throws for a value outside the i32 range. Shifting left and then
// arithmetically right by 32 reduces the value to its sign-extended low
// word, which always converts.

// CHECK: udonasm.push "__0_arg0__param"
// CHECK-NEXT: udonasm.push "[[BITS:[^"]+]]"
// CHECK-NEXT: udonasm.push "__vrc_i64_scratch_0"
// CHECK-NEXT: udonasm.extern "SystemInt64.__op_LeftShift__SystemInt64_SystemInt32__SystemInt64"
// CHECK-NEXT: udonasm.push "__vrc_i64_scratch_0"
// CHECK-NEXT: udonasm.push "[[BITS]]"
// CHECK-NEXT: udonasm.push "__vrc_i64_scratch_0"
// CHECK-NEXT: udonasm.extern "SystemInt64.__op_RightShift__SystemInt64_SystemInt32__SystemInt64"
// CHECK-NEXT: udonasm.push "__vrc_i64_scratch_0"
// CHECK-NEXT: udonasm.push "{{[^"]+}}"
// CHECK-NEXT: udonasm.extern "SystemConvert.__ToInt32__SystemInt64__SystemInt32"
func.func @low_word(%wide: i64) -> i32 attributes {vrc.export_id = 1 : i32} {
  %narrow = arith.trunci %wide : i64 to i32
  return %narrow : i32
}

// A value that was sign-extended from i32 is in range and converts directly.

// CHECK: udonasm.extern "SystemConvert.__ToInt64__SystemInt32__SystemInt64"
// CHECK-NEXT: udonasm.push "[[WIDE:[^"]+]]"
// CHECK-NEXT: udonasm.push "{{[^"]+}}"
// CHECK-NEXT: udonasm.extern "SystemConvert.__ToInt32__SystemInt64__SystemInt32"
func.func @round_trip(%value: i32) -> i32 attributes {vrc.export_id = 2 : i32} {
  %wide = arith.extsi %value : i32 to i64
  %narrow = arith.trunci %wide : i64 to i32
  return %narrow : i32
}
