// RUN: %vrc-opt --vrc-lower-udonhl-to-udonasm %s | %FileCheck %s

// Udon has only the arithmetic right shift of SystemInt32, which fills with
// the sign bit.

// It is the logical shift of a value whose sign bit is known to be clear,
// here a zero-extended byte.
// CHECK-DAG: {initial = "16777215", name = "[[MASK24:__vrc_constant_[0-9]+]]", type = "SystemInt32"}

// CHECK: udonasm.copy
// CHECK-NEXT: udonasm.push "[[BYTE:[^"]+]]"
// CHECK-NEXT: udonasm.push "__0_arg1__param"
// CHECK-NEXT: udonasm.push "[[RESULT:[^"]+]]"
// CHECK-NEXT: udonasm.extern "SystemInt32.__op_RightShift__SystemInt32_SystemInt32__SystemInt32"
// CHECK-NEXT: udonasm.push "[[RESULT]]"
// CHECK-NEXT: udonasm.push "__0___0_non_negative__ret"
func.func @non_negative(%value: i32, %amount: i32) -> i32 attributes {vrc.export_id = 1 : i32} {
  %byte = arith.trunci %value : i32 to i8
  %wide = arith.extui %byte : i8 to i32
  %shifted = arith.shrui %wide, %amount : i32
  return %shifted : i32
}

// A constant amount fixes the mask that clears the copies of the sign bit.
// CHECK: udonasm.push "__1_arg0__param"
// CHECK-NEXT: udonasm.push "{{[^"]+}}"
// CHECK-NEXT: udonasm.push "__vrc_lshr_i32_arithmetic"
// CHECK-NEXT: udonasm.extern "SystemInt32.__op_RightShift__SystemInt32_SystemInt32__SystemInt32"
// CHECK-NEXT: udonasm.push "__vrc_lshr_i32_arithmetic"
// CHECK-NEXT: udonasm.push "[[MASK24]]"
// CHECK-NEXT: udonasm.push "[[MASKED:[^"]+]]"
// CHECK-NEXT: udonasm.extern "SystemInt32.__op_LogicalAnd__SystemInt32_SystemInt32__SystemInt32"
// CHECK-NEXT: udonasm.push "[[MASKED]]"
// CHECK-NEXT: udonasm.push "__0___0_constant_amount__ret"
func.func @constant_amount(%value: i32) -> i32 attributes {vrc.export_id = 2 : i32} {
  %eight = arith.constant 8 : i32
  %shifted = arith.shrui %value, %eight : i32
  return %shifted : i32
}

// Otherwise the mask is computed from the run-time amount, and a zero
// amount, which would need a shift by 32 to build the mask, is a copy.
// CHECK: udonasm.extern "SystemInt32.__op_Equality__SystemInt32_SystemInt32__SystemBoolean"
// CHECK: udonasm.extern "SystemInt32.__op_RightShift__SystemInt32_SystemInt32__SystemInt32"
// CHECK: udonasm.extern "SystemInt32.__op_LeftShift__SystemInt32_SystemInt32__SystemInt32"
// CHECK: udonasm.extern "SystemInt32.__op_LogicalXor__SystemInt32_SystemInt32__SystemInt32"
// CHECK: udonasm.extern "SystemInt32.__op_LogicalAnd__SystemInt32_SystemInt32__SystemInt32"
func.func @unknown(%value: i32, %amount: i32) -> i32 attributes {vrc.export_id = 3 : i32} {
  %shifted = arith.shrui %value, %amount : i32
  return %shifted : i32
}
