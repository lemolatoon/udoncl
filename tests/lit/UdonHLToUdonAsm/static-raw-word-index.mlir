// RUN: %vrc-opt --vrc-lower-udonhl-to-udonasm %s | %FileCheck %s

// The raw heap layout is planned by this lowering, so the address of a raw
// global or alloca, and any constant byte offset from one, is known here. A
// word-aligned access through such an address indexes the raw heap with a
// literal word index instead of shifting the byte address at run time, and an
// address whose every use is such an access is never materialized.

"udonhl.global"() <{sym_name = "cell", storage_type = !udonhl.ptr<0>, initial_value = 0 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> : () -> ()
"udonhl.global"() <{sym_name = "pair", storage_type = !udonhl.ptr<0>, initial_value = 0 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> : () -> ()

// @cell is the word 1 and @pair the word 2 of the raw heap.
// CHECK-DAG: {initial = "1", name = "[[CELL:__vrc_constant_[0-9]+]]", type = "SystemInt32"}
// CHECK-DAG: {initial = "2", name = "[[PAIR:__vrc_constant_[0-9]+]]", type = "SystemInt32"}
// CHECK-DAG: {initial = "6", name = "[[LOW:__vrc_constant_[0-9]+]]", type = "SystemInt32"}
// CHECK-DAG: {initial = "7", name = "[[HIGH:__vrc_constant_[0-9]+]]", type = "SystemInt32"}

// Load, increment and store through the folded address: no byte address is
// built and no index is shifted.
// CHECK: udonasm.copy
// CHECK-NEXT: udonasm.push "__vrc_raw_heap"
// CHECK-NEXT: udonasm.push "[[CELL]]"
// CHECK-NEXT: udonasm.push "[[LOADED:[^"]+]]"
// CHECK-NEXT: udonasm.extern "SystemInt32Array.__Get__SystemInt32__SystemInt32"
// CHECK-NEXT: udonasm.push "[[LOADED]]"
// CHECK-NEXT: udonasm.push "{{[^"]+}}"
// CHECK-NEXT: udonasm.push "[[NEXT:[^"]+]]"
// CHECK-NEXT: udonasm.extern "SystemInt32.__op_Addition__SystemInt32_SystemInt32__SystemInt32"
// CHECK-NEXT: udonasm.push "__vrc_raw_heap"
// CHECK-NEXT: udonasm.push "[[CELL]]"
// CHECK-NEXT: udonasm.push "[[NEXT]]"
// CHECK-NEXT: udonasm.extern "SystemInt32Array.__Set__SystemInt32_SystemInt32__SystemVoid"
// CHECK-NEXT: udonasm.push "[[NEXT]]"
// CHECK-NEXT: udonasm.push "__0_increment__ret"
func.func @increment() -> i32 attributes {vrc.export_id = 1 : i32} {
  %address = udonhl.address_of @cell : !udonhl.ptr<0>
  %loaded = "udonhl.load"(%address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> i32
  %one = arith.constant 1 : i32
  %next = arith.addi %loaded, %one : i32
  "udonhl.store"(%next, %address) <{alignment = 4 : i64}> : (i32, !udonhl.ptr<0>) -> ()
  return %next : i32
}

// An address with any other use, here a run-time offset, is still
// materialized, and the access through the derived address still shifts. The
// access through the address itself is folded all the same.
// CHECK: udonasm.push "[[PAIR]]"
// CHECK-NEXT: udonasm.push "{{[^"]+}}"
// CHECK-NEXT: udonasm.extern "SystemInt32Array.__Set__SystemInt32_SystemInt32__SystemVoid"
// CHECK-NEXT: udonasm.push "[[BASE:[^"]+]]"
// CHECK-NEXT: udonasm.push "{{[^"]+}}"
// CHECK-NEXT: udonasm.push "[[DERIVED:[^"]+]]"
// CHECK-NEXT: udonasm.extern "SystemInt32.__op_Addition__SystemInt32_SystemInt32__SystemInt32"
// CHECK-NEXT: udonasm.push "[[DERIVED]]"
// CHECK-NEXT: udonasm.push "{{[^"]+}}"
// CHECK-NEXT: udonasm.push "__vrc_raw_load_word_index"
// CHECK-NEXT: udonasm.extern "SystemInt32.__op_RightShift__SystemInt32_SystemInt32__SystemInt32"
func.func @indexed(%value: i32, %offset: i32) -> i32 attributes {vrc.export_id = 2 : i32} {
  %address = udonhl.address_of @pair : !udonhl.ptr<0>
  "udonhl.store"(%value, %address) <{alignment = 4 : i64}> : (i32, !udonhl.ptr<0>) -> ()
  %derived = "udonhl.ptr_add"(%address, %offset) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
  %loaded = "udonhl.load"(%derived) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> i32
  return %loaded : i32
}

// A constant offset from a raw alloca folds too, and both words of an
// aligned i64 are literal indices: the alloca starts at the word 4 and its
// second i64 is the words 6 and 7.
// CHECK: udonasm.push "[[LOW]]"
// CHECK-NEXT: udonasm.push "__vrc_raw_load_i64_low"
// CHECK-NEXT: udonasm.extern "SystemInt32Array.__Get__SystemInt32__SystemInt32"
// CHECK-NEXT: udonasm.push "__vrc_raw_heap"
// CHECK-NEXT: udonasm.push "[[HIGH]]"
// CHECK-NEXT: udonasm.push "__vrc_raw_load_i64_high"
// CHECK-NEXT: udonasm.extern "SystemInt32Array.__Get__SystemInt32__SystemInt32"
func.func @second_wide() -> i64 attributes {vrc.export_id = 3 : i32} {
  %bytes = arith.constant 16 : i32
  %base = "udonhl.alloca"(%bytes) <{alignment = 8 : i64}> : (i32) -> !udonhl.ptr<0>
  %eight = arith.constant 8 : i32
  %second = "udonhl.ptr_add"(%base, %eight) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
  %loaded = "udonhl.load"(%second) <{alignment = 8 : i64}> : (!udonhl.ptr<0>) -> i64
  return %loaded : i64
}
