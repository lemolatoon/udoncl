// RUN: %vrc-opt --vrc-lower-udonhl-to-udonasm %s | %FileCheck %s

// The Worlds SDK assembler has no SystemInt64 literal, so an i64 constant is
// built at its use from SystemInt32 literal seeds. A sign-extended word is one
// SystemConvert.ToInt64; a shifted word adds one shift; any other value is
// its two words, with the high word complemented when the low word is
// negative so that the exclusive OR of the sign-extended low word restores
// it.

// CHECK-DAG: {initial = "-4", name = "[[SMALL:__vrc_constant_[0-9]+]]", type = "SystemInt32"}
// CHECK-DAG: {initial = "-1", name = "[[ONES:__vrc_constant_[0-9]+]]", type = "SystemInt32"}
// CHECK-DAG: {initial = "63", name = "[[SIGN_SHIFT:__vrc_constant_[0-9]+]]", type = "SystemInt32"}
// CHECK-DAG: {initial = "32", name = "[[WORD_SHIFT:__vrc_constant_[0-9]+]]", type = "SystemInt32"}
// CHECK-DAG: {initial = "-305419897", name = "[[HIGH:__vrc_constant_[0-9]+]]", type = "SystemInt32"}
// CHECK-DAG: {initial = "-1698898192", name = "[[LOW:__vrc_constant_[0-9]+]]", type = "SystemInt32"}

// CHECK: udonasm.push "[[SMALL]]"
// CHECK-NEXT: udonasm.push "[[SMALL_RESULT:[^"]+]]"
// CHECK-NEXT: udonasm.extern "SystemConvert.__ToInt64__SystemInt32__SystemInt64"
// CHECK-NEXT: udonasm.push "[[SMALL_RESULT]]"
// CHECK-NEXT: udonasm.push "__0_small__ret"
func.func @small() -> i64 attributes {vrc.export_id = 1 : i32} {
  %value = arith.constant -4 : i64
  return %value : i64
}

// CHECK: udonasm.push "[[ONES]]"
// CHECK-NEXT: udonasm.push "[[SIGN_RESULT:[^"]+]]"
// CHECK-NEXT: udonasm.extern "SystemConvert.__ToInt64__SystemInt32__SystemInt64"
// CHECK-NEXT: udonasm.push "[[SIGN_RESULT]]"
// CHECK-NEXT: udonasm.push "[[SIGN_SHIFT]]"
// CHECK-NEXT: udonasm.push "[[SIGN_RESULT]]"
// CHECK-NEXT: udonasm.extern "SystemInt64.__op_LeftShift__SystemInt64_SystemInt32__SystemInt64"
// CHECK-NEXT: udonasm.push "[[SIGN_RESULT]]"
// CHECK-NEXT: udonasm.push "__0_sign_mask__ret"
func.func @sign_mask() -> i64 attributes {vrc.export_id = 2 : i32} {
  %value = arith.constant -9223372036854775808 : i64
  return %value : i64
}

// 0x123456789ABCDEF0: the low word is negative, the high word complemented.
// CHECK: udonasm.push "[[HIGH]]"
// CHECK-NEXT: udonasm.push "[[WIDE_RESULT:[^"]+]]"
// CHECK-NEXT: udonasm.extern "SystemConvert.__ToInt64__SystemInt32__SystemInt64"
// CHECK-NEXT: udonasm.push "[[WIDE_RESULT]]"
// CHECK-NEXT: udonasm.push "[[WORD_SHIFT]]"
// CHECK-NEXT: udonasm.push "[[WIDE_RESULT]]"
// CHECK-NEXT: udonasm.extern "SystemInt64.__op_LeftShift__SystemInt64_SystemInt32__SystemInt64"
// CHECK-NEXT: udonasm.push "[[LOW]]"
// CHECK-NEXT: udonasm.push "__vrc_i64_scratch_0"
// CHECK-NEXT: udonasm.extern "SystemConvert.__ToInt64__SystemInt32__SystemInt64"
// CHECK-NEXT: udonasm.push "[[WIDE_RESULT]]"
// CHECK-NEXT: udonasm.push "__vrc_i64_scratch_0"
// CHECK-NEXT: udonasm.push "[[WIDE_RESULT]]"
// CHECK-NEXT: udonasm.extern "SystemInt64.__op_LogicalXor__SystemInt64_SystemInt64__SystemInt64"
// CHECK-NEXT: udonasm.push "[[WIDE_RESULT]]"
// CHECK-NEXT: udonasm.push "__0_two_words__ret"
func.func @two_words() -> i64 attributes {vrc.export_id = 3 : i32} {
  %value = arith.constant 1311768467463790320 : i64
  return %value : i64
}

// 0x7000000180000000: the low word is the minimum SystemInt32, which has no
// literal seed, so the three non-zero bytes are assembled instead.
// CHECK: udonasm.extern "SystemConvert.__ToInt64__SystemInt32__SystemInt64"
// CHECK: udonasm.extern "SystemInt64.__op_LeftShift__SystemInt64_SystemInt32__SystemInt64"
// CHECK: udonasm.extern "SystemInt64.__op_LogicalOr__SystemInt64_SystemInt64__SystemInt64"
// CHECK: udonasm.extern "SystemInt64.__op_LogicalOr__SystemInt64_SystemInt64__SystemInt64"
// CHECK-NOT: udonasm.extern "SystemInt64.__op_LogicalOr__SystemInt64_SystemInt64__SystemInt64"
// CHECK: udonasm.push "__0_minimum_low_word__ret"
func.func @minimum_low_word() -> i64 attributes {vrc.export_id = 4 : i32} {
  %value = arith.constant 8070450538690379776 : i64
  return %value : i64
}
