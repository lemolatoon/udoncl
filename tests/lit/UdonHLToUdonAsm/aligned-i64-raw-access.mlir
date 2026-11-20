// RUN: %vrc-opt --vrc-lower-udonhl-to-udonasm %s | %FileCheck %s

// A word-aligned i64 raw access is two words of the SystemInt32 raw heap, not
// eight packed bytes.
//
// The store reduces each word to its sign-extended 32 bits by arithmetic
// shifts, because SystemConvert.ToInt32 is range checked.
//
// The load widens both words. Widening sign-extends the low word, which flips
// every high bit when it is negative, so the high word is first complemented
// by the low word's sign and the exclusive OR restores it.

// CHECK-DAG: {initial = "1", name = "[[ONE:__vrc_constant_[0-9]+]]", type = "SystemInt32"}
// CHECK-DAG: {initial = "2", name = "[[TWO:__vrc_constant_[0-9]+]]", type = "SystemInt32"}
// CHECK-DAG: {initial = "31", name = "[[SIGN:__vrc_constant_[0-9]+]]", type = "SystemInt32"}
// CHECK-DAG: {initial = "32", name = "[[WORD:__vrc_constant_[0-9]+]]", type = "SystemInt32"}

// CHECK: udonasm.push "[[TWO]]"
// CHECK-NEXT: udonasm.push "__vrc_raw_store_word_index"
// CHECK-NEXT: udonasm.extern "SystemInt32.__op_RightShift__SystemInt32_SystemInt32__SystemInt32"
// CHECK-NEXT: udonasm.push "[[VALUE:[^"]+]]"
// CHECK-NEXT: udonasm.push "[[WORD]]"
// CHECK-NEXT: udonasm.push "__vrc_i64_scratch_0"
// CHECK-NEXT: udonasm.extern "SystemInt64.__op_LeftShift__SystemInt64_SystemInt32__SystemInt64"
// CHECK-NEXT: udonasm.push "__vrc_i64_scratch_0"
// CHECK-NEXT: udonasm.push "[[WORD]]"
// CHECK-NEXT: udonasm.push "__vrc_i64_scratch_0"
// CHECK-NEXT: udonasm.extern "SystemInt64.__op_RightShift__SystemInt64_SystemInt32__SystemInt64"
// CHECK-NEXT: udonasm.push "__vrc_i64_scratch_0"
// CHECK-NEXT: udonasm.push "__vrc_raw_store_word"
// CHECK-NEXT: udonasm.extern "SystemConvert.__ToInt32__SystemInt64__SystemInt32"
// CHECK-NEXT: udonasm.push "__vrc_raw_heap"
// CHECK-NEXT: udonasm.push "__vrc_raw_store_word_index"
// CHECK-NEXT: udonasm.push "__vrc_raw_store_word"
// CHECK-NEXT: udonasm.extern "SystemInt32Array.__Set__SystemInt32_SystemInt32__SystemVoid"
// CHECK-NEXT: udonasm.push "__vrc_raw_store_word_index"
// CHECK-NEXT: udonasm.push "[[ONE]]"
// CHECK-NEXT: udonasm.push "__vrc_raw_store_word_index"
// CHECK-NEXT: udonasm.extern "SystemInt32.__op_Addition__SystemInt32_SystemInt32__SystemInt32"
// CHECK-NEXT: udonasm.push "[[VALUE]]"
// CHECK-NEXT: udonasm.push "[[WORD]]"
// CHECK-NEXT: udonasm.push "__vrc_i64_scratch_0"
// CHECK-NEXT: udonasm.extern "SystemInt64.__op_RightShift__SystemInt64_SystemInt32__SystemInt64"
// CHECK-NEXT: udonasm.push "__vrc_i64_scratch_0"
// CHECK-NEXT: udonasm.push "__vrc_raw_store_word"
// CHECK-NEXT: udonasm.extern "SystemConvert.__ToInt32__SystemInt64__SystemInt32"
// CHECK-NEXT: udonasm.push "__vrc_raw_heap"
// CHECK-NEXT: udonasm.push "__vrc_raw_store_word_index"
// CHECK-NEXT: udonasm.push "__vrc_raw_store_word"
// CHECK-NEXT: udonasm.extern "SystemInt32Array.__Set__SystemInt32_SystemInt32__SystemVoid"

// CHECK-NEXT: udonasm.push "{{[^"]+}}"
// CHECK-NEXT: udonasm.push "[[TWO]]"
// CHECK-NEXT: udonasm.push "__vrc_raw_load_word_index"
// CHECK-NEXT: udonasm.extern "SystemInt32.__op_RightShift__SystemInt32_SystemInt32__SystemInt32"
// CHECK-NEXT: udonasm.push "__vrc_raw_heap"
// CHECK-NEXT: udonasm.push "__vrc_raw_load_word_index"
// CHECK-NEXT: udonasm.push "__vrc_raw_load_i64_low"
// CHECK-NEXT: udonasm.extern "SystemInt32Array.__Get__SystemInt32__SystemInt32"
// CHECK-NEXT: udonasm.push "__vrc_raw_load_word_index"
// CHECK-NEXT: udonasm.push "[[ONE]]"
// CHECK-NEXT: udonasm.push "__vrc_raw_load_word_index"
// CHECK-NEXT: udonasm.extern "SystemInt32.__op_Addition__SystemInt32_SystemInt32__SystemInt32"
// CHECK-NEXT: udonasm.push "__vrc_raw_heap"
// CHECK-NEXT: udonasm.push "__vrc_raw_load_word_index"
// CHECK-NEXT: udonasm.push "__vrc_raw_load_i64_high"
// CHECK-NEXT: udonasm.extern "SystemInt32Array.__Get__SystemInt32__SystemInt32"
// CHECK-NEXT: udonasm.push "__vrc_raw_load_i64_low"
// CHECK-NEXT: udonasm.push "[[SIGN]]"
// CHECK-NEXT: udonasm.push "__vrc_raw_load_i64_low_sign"
// CHECK-NEXT: udonasm.extern "SystemInt32.__op_RightShift__SystemInt32_SystemInt32__SystemInt32"
// CHECK-NEXT: udonasm.push "__vrc_raw_load_i64_high"
// CHECK-NEXT: udonasm.push "__vrc_raw_load_i64_low_sign"
// CHECK-NEXT: udonasm.push "__vrc_raw_load_i64_high"
// CHECK-NEXT: udonasm.extern "SystemInt32.__op_LogicalXor__SystemInt32_SystemInt32__SystemInt32"
// CHECK-NEXT: udonasm.push "__vrc_raw_load_i64_high"
// CHECK-NEXT: udonasm.push "[[LOADED:[^"]+]]"
// CHECK-NEXT: udonasm.extern "SystemConvert.__ToInt64__SystemInt32__SystemInt64"
// CHECK-NEXT: udonasm.push "[[LOADED]]"
// CHECK-NEXT: udonasm.push "[[WORD]]"
// CHECK-NEXT: udonasm.push "[[LOADED]]"
// CHECK-NEXT: udonasm.extern "SystemInt64.__op_LeftShift__SystemInt64_SystemInt32__SystemInt64"
// CHECK-NEXT: udonasm.push "__vrc_raw_load_i64_low"
// CHECK-NEXT: udonasm.push "__vrc_i64_scratch_0"
// CHECK-NEXT: udonasm.extern "SystemConvert.__ToInt64__SystemInt32__SystemInt64"
// CHECK-NEXT: udonasm.push "[[LOADED]]"
// CHECK-NEXT: udonasm.push "__vrc_i64_scratch_0"
// CHECK-NEXT: udonasm.push "[[LOADED]]"
// CHECK-NEXT: udonasm.extern "SystemInt64.__op_LogicalXor__SystemInt64_SystemInt64__SystemInt64"
func.func @aligned(%value: i64, %address: i32) -> i64 attributes {vrc.export_id = 1 : i32} {
  %pointer = "udonhl.inttoptr"(%address) : (i32) -> !udonhl.ptr<0>
  "udonhl.store"(%value, %pointer) <{alignment = 8 : i64}> : (i64, !udonhl.ptr<0>) -> ()
  %loaded = "udonhl.load"(%pointer) <{alignment = 8 : i64}> : (!udonhl.ptr<0>) -> i64
  return %loaded : i64
}

// An access that is not word aligned still goes through packed bytes.
// CHECK: udonasm.push "__vrc_raw_byte_store_word_index"
// CHECK: udonasm.push "__vrc_raw_byte_load_word_index"
func.func @packed(%value: i64, %address: i32) -> i64 attributes {vrc.export_id = 2 : i32} {
  %pointer = "udonhl.inttoptr"(%address) : (i32) -> !udonhl.ptr<0>
  "udonhl.store"(%value, %pointer) <{alignment = 1 : i64}> : (i64, !udonhl.ptr<0>) -> ()
  %loaded = "udonhl.load"(%pointer) <{alignment = 1 : i64}> : (!udonhl.ptr<0>) -> i64
  return %loaded : i64
}
