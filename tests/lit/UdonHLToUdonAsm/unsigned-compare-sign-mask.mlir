// RUN: %vrc-opt --vrc-lower-udonhl-to-udonasm %s | %FileCheck %s

// An unsigned comparison is the signed comparison of both operands with the
// sign bit flipped. The assembler cannot spell the minimum SystemInt32, so
// the i32 sign mask is one shift of the literal 1 and the i64 sign mask is one
// widened literal and one shift.

// CHECK-DAG: {initial = "1", name = "[[ONE:__vrc_constant_[0-9]+]]", type = "SystemInt32"}
// CHECK-DAG: {initial = "31", name = "[[WORD_SIGN:__vrc_constant_[0-9]+]]", type = "SystemInt32"}
// CHECK-DAG: {initial = "-1", name = "[[ONES:__vrc_constant_[0-9]+]]", type = "SystemInt32"}
// CHECK-DAG: {initial = "63", name = "[[WIDE_SIGN:__vrc_constant_[0-9]+]]", type = "SystemInt32"}

// CHECK: udonasm.push "[[ONE]]"
// CHECK-NEXT: udonasm.push "[[WORD_SIGN]]"
// CHECK-NEXT: udonasm.push "__vrc_unsigned_cmp_sign_i32"
// CHECK-NEXT: udonasm.extern "SystemInt32.__op_LeftShift__SystemInt32_SystemInt32__SystemInt32"
// CHECK-NEXT: udonasm.push "{{[^"]+}}"
// CHECK-NEXT: udonasm.push "__vrc_unsigned_cmp_sign_i32"
// CHECK-NEXT: udonasm.push "__vrc_unsigned_cmp_lhs_i32"
// CHECK-NEXT: udonasm.extern "SystemInt32.__op_LogicalXor__SystemInt32_SystemInt32__SystemInt32"
// CHECK-NEXT: udonasm.push "{{[^"]+}}"
// CHECK-NEXT: udonasm.push "__vrc_unsigned_cmp_sign_i32"
// CHECK-NEXT: udonasm.push "__vrc_unsigned_cmp_rhs_i32"
// CHECK-NEXT: udonasm.extern "SystemInt32.__op_LogicalXor__SystemInt32_SystemInt32__SystemInt32"
// CHECK-NEXT: udonasm.push "__vrc_unsigned_cmp_lhs_i32"
// CHECK-NEXT: udonasm.push "__vrc_unsigned_cmp_rhs_i32"
// CHECK-NEXT: udonasm.push "{{[^"]+}}"
// CHECK-NEXT: udonasm.extern "SystemInt32.__op_LessThan__SystemInt32_SystemInt32__SystemBoolean"
func.func @below(%lhs: i32, %rhs: i32) -> i1 attributes {vrc.export_id = 1 : i32} {
  %result = arith.cmpi ult, %lhs, %rhs : i32
  return %result : i1
}

// CHECK: udonasm.push "[[ONES]]"
// CHECK-NEXT: udonasm.push "__vrc_unsigned_cmp_sign_i64"
// CHECK-NEXT: udonasm.extern "SystemConvert.__ToInt64__SystemInt32__SystemInt64"
// CHECK-NEXT: udonasm.push "__vrc_unsigned_cmp_sign_i64"
// CHECK-NEXT: udonasm.push "[[WIDE_SIGN]]"
// CHECK-NEXT: udonasm.push "__vrc_unsigned_cmp_sign_i64"
// CHECK-NEXT: udonasm.extern "SystemInt64.__op_LeftShift__SystemInt64_SystemInt32__SystemInt64"
// CHECK-NEXT: udonasm.push "{{[^"]+}}"
// CHECK-NEXT: udonasm.push "__vrc_unsigned_cmp_sign_i64"
// CHECK-NEXT: udonasm.push "__vrc_unsigned_cmp_lhs_i64"
// CHECK-NEXT: udonasm.extern "SystemInt64.__op_LogicalXor__SystemInt64_SystemInt64__SystemInt64"
func.func @wide_above(%lhs: i64, %rhs: i64) -> i1 attributes {vrc.export_id = 2 : i32} {
  %result = arith.cmpi ugt, %lhs, %rhs : i64
  return %result : i1
}
