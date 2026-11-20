// RUN: %vrc-opt --split-input-file --vrc-evaluate-host-constructors %s | %FileCheck %s

// A static constructor runs at build time. What it stored becomes the
// initializer of each global it wrote, and its body shrinks to its result.

// Integer semantics are the wrapping two's-complement ones of the target.
// CHECK: initial_value = -2147483648 : i32
// CHECK-SAME: sym_name = "wrapped"
"udonhl.global"() <{sym_name = "wrapped", storage_type = i32, initial_value = 0 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> : () -> ()
// 0x100000000 * 3 + 5, low word first.
// CHECK: initial_value = array<i8: 5, 0, 0, 0, 3, 0, 0, 0>
// CHECK-SAME: sym_name = "wide"
"udonhl.global"() <{sym_name = "wide", storage_type = !llvm.array<8 x i8>, initial_value = array<i8: 0, 0, 0, 0, 0, 0, 0, 0>, linkage = "internal", address_space = 0 : i32, alignment = 8 : i64}> : () -> ()
// 1 + 2 + ... + 10 through a loop and a call.
// CHECK: initial_value = 55 : i32
// CHECK-SAME: sym_name = "sum"
"udonhl.global"() <{sym_name = "sum", storage_type = i32, initial_value = 0 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> : () -> ()
// sext(trunc(0x1ff) : i8) = -1, lshr -> 0x0fffffff, ashr -> -1, and the
// switch selects 30 for the flag 3.
// CHECK: initial_value = array<i8: -1, -1, -1, -1, -1, -1, -1, 15, -1, -1, -1, -1, 30, 0, 0, 0>
// CHECK-SAME: sym_name = "casts"
"udonhl.global"() <{sym_name = "casts", storage_type = !llvm.array<16 x i8>, initial_value = array<i8: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> : () -> ()
// An untouched global keeps its initializer.
// CHECK: initial_value = 7 : i32
// CHECK-SAME: sym_name = "untouched"
"udonhl.global"() <{sym_name = "untouched", storage_type = i32, initial_value = 7 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> : () -> ()

func.func private @add_up_to(%limit: i32) -> i32 {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  cf.br ^header(%one, %zero : i32, i32)
^header(%index: i32, %total: i32):
  %more = arith.cmpi sle, %index, %limit : i32
  cf.cond_br %more, ^body, ^exit
^body:
  %next_total = arith.addi %total, %index : i32
  %next_index = arith.addi %index, %one : i32
  cf.br ^header(%next_index, %next_total : i32, i32)
^exit:
  return %total : i32
}

// CHECK-LABEL: func.func @integers() -> i32
// CHECK-SAME: vrc.evaluated_static_constructor = {evaluated_operations = {{[0-9]+}} : i64, residual_commands = 0 : i32}
// CHECK-NOT: vrc.static_constructor
// CHECK-NEXT: %[[STATUS:.*]] = arith.constant 0 : i32
// CHECK-NEXT: return %[[STATUS]] : i32
// CHECK-NEXT: }
func.func @integers() -> i32 attributes {vrc.static_constructor} {
  %max = arith.constant 2147483647 : i32
  %one = arith.constant 1 : i32
  %overflow = arith.addi %max, %one overflow<nsw> : i32
  %wrapped = udonhl.address_of @wrapped : !udonhl.ptr<0>
  "udonhl.store"(%overflow, %wrapped) <{alignment = 4 : i64}> : (i32, !udonhl.ptr<0>) -> ()

  %high = arith.constant 4294967296 : i64
  %three = arith.constant 3 : i64
  %five = arith.constant 5 : i64
  %product = arith.muli %high, %three : i64
  %wide_value = arith.addi %product, %five : i64
  %wide = udonhl.address_of @wide : !udonhl.ptr<0>
  "udonhl.store"(%wide_value, %wide) <{alignment = 8 : i64}> : (i64, !udonhl.ptr<0>) -> ()

  %ten = arith.constant 10 : i32
  %total = call @add_up_to(%ten) : (i32) -> i32
  %sum = udonhl.address_of @sum : !udonhl.ptr<0>
  "udonhl.store"(%total, %sum) <{alignment = 4 : i64}> : (i32, !udonhl.ptr<0>) -> ()

  %casts = udonhl.address_of @casts : !udonhl.ptr<0>
  %bits = arith.constant 511 : i32
  %narrow = arith.trunci %bits : i32 to i8
  %signed = arith.extsi %narrow : i8 to i32
  "udonhl.store"(%signed, %casts) <{alignment = 4 : i64}> : (i32, !udonhl.ptr<0>) -> ()
  %four = arith.constant 4 : i32
  %logical = arith.shrui %signed, %four : i32
  %logical_address = "udonhl.ptr_add"(%casts, %four) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
  "udonhl.store"(%logical, %logical_address) <{alignment = 4 : i64}> : (i32, !udonhl.ptr<0>) -> ()
  %arithmetic = arith.shrsi %signed, %four : i32
  %eight = arith.constant 8 : i32
  %arithmetic_address = "udonhl.ptr_add"(%casts, %eight) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
  "udonhl.store"(%arithmetic, %arithmetic_address) <{alignment = 4 : i64}> : (i32, !udonhl.ptr<0>) -> ()
  %flag = arith.constant 3 : i32
  cf.switch %flag : i32, [
    default: ^selected(%one : i32),
    2: ^selected(%ten : i32),
    3: ^thirty
  ]
^thirty:
  %thirty = arith.constant 30 : i32
  cf.br ^selected(%thirty : i32)
^selected(%choice: i32):
  %twelve = arith.constant 12 : i32
  %choice_address = "udonhl.ptr_add"(%casts, %twelve) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
  "udonhl.store"(%choice, %choice_address) <{alignment = 4 : i64}> : (i32, !udonhl.ptr<0>) -> ()
  %status = arith.constant 0 : i32
  return %status : i32
}

// -----

// Byte memory is little-endian and byte-addressed: sub-word and unaligned
// accesses, fills and copies behave as the raw heap does. Automatic storage
// is scratch and is not committed.

// memset(8 bytes, 0xAA), then an unaligned i16 0x0102 at byte 3 and a
// boolean at byte 7.
// CHECK: initial_value = array<i8: -86, -86, -86, 2, 1, -86, -86, 1>
// CHECK-SAME: sym_name = "bytes"
"udonhl.global"() <{sym_name = "bytes", storage_type = !llvm.array<8 x i8>, initial_value = array<i8: 0, 0, 0, 0, 0, 0, 0, 0>, linkage = "internal", address_space = 0 : i32, alignment = 1 : i64}> : () -> ()
// A copy of the bytes 2..5 of @bytes through an alloca.
// CHECK: initial_value = array<i8: -86, 2, 1, -86>
// CHECK-SAME: sym_name = "copy"
"udonhl.global"() <{sym_name = "copy", storage_type = !llvm.array<4 x i8>, initial_value = array<i8: 9, 9, 9, 9>, linkage = "internal", address_space = 0 : i32, alignment = 1 : i64}> : () -> ()
// The boolean read back from byte 7 selects 1.5.
// CHECK: initial_value = 1.500000e+00 : f32
// CHECK-SAME: sym_name = "real"
"udonhl.global"() <{sym_name = "real", storage_type = f32, initial_value = 0.0 : f32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> : () -> ()
// Element 2 of a typed Udon array.
// CHECK: memref.global "private" @table : memref<4xi32> = dense<[0, 0, 42, 0]>
memref.global "private" @table : memref<4xi32> = dense<0>

// CHECK-LABEL: func.func @memory()
// CHECK-NEXT: return
func.func @memory() attributes {vrc.static_constructor} {
  %bytes = udonhl.address_of @bytes : !udonhl.ptr<0>
  %fill = arith.constant -86 : i8
  %eight = arith.constant 8 : i32
  "udonhl.memset"(%bytes, %fill, %eight) : (!udonhl.ptr<0>, i8, i32) -> ()
  %three = arith.constant 3 : i32
  %unaligned = "udonhl.ptr_add"(%bytes, %three) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
  %short = arith.constant 258 : i16
  "udonhl.store"(%short, %unaligned) <{alignment = 1 : i64}> : (i16, !udonhl.ptr<0>) -> ()
  %seven = arith.constant 7 : i32
  %last = "udonhl.ptr_add"(%bytes, %seven) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
  %true = arith.constant true
  "udonhl.store"(%true, %last) <{alignment = 1 : i64}> : (i1, !udonhl.ptr<0>) -> ()

  %four = arith.constant 4 : i32
  %scratch = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
  %two = arith.constant 2 : i32
  %middle = "udonhl.ptr_add"(%bytes, %two) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
  "udonhl.memcpy"(%scratch, %middle, %four) : (!udonhl.ptr<0>, !udonhl.ptr<0>, i32) -> ()
  %copy = udonhl.address_of @copy : !udonhl.ptr<0>
  "udonhl.memmove"(%copy, %scratch, %four) : (!udonhl.ptr<0>, !udonhl.ptr<0>, i32) -> ()

  %flag = "udonhl.load"(%last) <{alignment = 1 : i64}> : (!udonhl.ptr<0>) -> i1
  %yes = arith.constant 1.5 : f32
  %no = arith.constant 2.5 : f32
  %chosen = arith.select %flag, %yes, %no : f32
  %real = udonhl.address_of @real : !udonhl.ptr<0>
  "udonhl.store"(%chosen, %real) <{alignment = 4 : i64}> : (f32, !udonhl.ptr<0>) -> ()

  %table = memref.get_global @table : memref<4xi32>
  %answer = arith.constant 42 : i32
  %index = arith.index_cast %two : i32 to index
  memref.store %answer, %table[%index] : memref<4xi32>
  return
}

// -----

// An address is symbolic until the raw heap is laid out: a stored address
// becomes a relocation, and so does the token of an assigned Unity buffer.

// CHECK: initial_value = 0 : i32
// CHECK-SAME: relocations = [{addend = 0 : i32, offset = 0 : i32, symbol = @record}]
// CHECK-SAME: sym_name = "head"
"udonhl.global"() <{sym_name = "head", storage_type = !udonhl.ptr<0>, initial_value = 0 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> : () -> ()
// {next = &record + 8 (a field of itself), token, value = 11 read back
// through @head, flags}. An alignment of 1 cannot hold a relocation.
// CHECK: alignment = 4 : i64
// CHECK-SAME: initial_value = array<i8: 0, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 13, 0, 0, 0>
// CHECK-SAME: relocations = [{addend = 8 : i32, offset = 0 : i32, symbol = @record}, {assigned_buffer = {capacity_bytes = 64 : i64, format = "rgba8_unorm", resource_id = 2 : i32, texture_id = 5 : i32}, offset = 4 : i32}]
// CHECK-SAME: sym_name = "record"
"udonhl.global"() <{sym_name = "record", storage_type = !llvm.array<16 x i8>, initial_value = array<i8: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>, linkage = "internal", address_space = 0 : i32, alignment = 1 : i64}> : () -> ()
// A relocation the constructor overwrites with null is dropped.
// CHECK: initial_value = 0 : i32
// CHECK-NOT: relocations
// CHECK-SAME: sym_name = "cleared"
"udonhl.global"() <{sym_name = "cleared", storage_type = !udonhl.ptr<0>, initial_value = 0 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64, relocations = [{offset = 0 : i32, symbol = @record, addend = 0 : i32}]}> : () -> ()
// A pointer cell that holds a non-address word (the odd tag of a static
// tensor handle) is an integer in memory.
// CHECK: initial_value = 7 : i32
// CHECK-SAME: storage_type = i32
// CHECK-SAME: sym_name = "tag"
"udonhl.global"() <{sym_name = "tag", storage_type = !udonhl.ptr<0>, initial_value = 0 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> : () -> ()

// CHECK-LABEL: func.func @addresses() -> i32
// CHECK-NEXT: %[[STATUS:.*]] = arith.constant 0 : i32
// CHECK-NEXT: return %[[STATUS]] : i32
func.func @addresses() -> i32 attributes {vrc.static_constructor} {
  %zero = arith.constant 0 : i32
  %null = "udonhl.inttoptr"(%zero) : (i32) -> !udonhl.ptr<0>
  %head = udonhl.address_of @head : !udonhl.ptr<0>
  %record = udonhl.address_of @record : !udonhl.ptr<0>
  "udonhl.store"(%record, %head) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()

  // record.value = 11, written through the address loaded from @head.
  %loaded = "udonhl.load"(%head) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
  %eight = arith.constant 8 : i32
  %value = "udonhl.ptr_add"(%loaded, %eight) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
  %eleven = arith.constant 11 : i32
  "udonhl.store"(%eleven, %value) <{alignment = 4 : i64}> : (i32, !udonhl.ptr<0>) -> ()
  // record.next = &record.value, through an integer round trip.
  %integer = "udonhl.ptrtoint"(%value) : (!udonhl.ptr<0>) -> i32
  %pointer = "udonhl.inttoptr"(%integer) : (i32) -> !udonhl.ptr<0>
  "udonhl.store"(%pointer, %record) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()

  %buffer = "vrc.assigned_buffer"() <{capacity_bytes = 64 : i64, format = "rgba8_unorm", resource_id = 2 : i32, texture_id = 5 : i32}> : () -> !vrc.buffer
  %token = builtin.unrealized_conversion_cast %buffer : !vrc.buffer to i32
  %four = arith.constant 4 : i32
  %token_address = "udonhl.ptr_add"(%record, %four) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
  "udonhl.store"(%token, %token_address) <{alignment = 4 : i64}> : (i32, !udonhl.ptr<0>) -> ()

  // Facts about addresses that hold in every layout: an object is not null,
  // equal addresses compare equal, addresses inside distinct objects differ,
  // a token of an assigned buffer is not zero, the low bits of an aligned
  // base are zero, and two addresses in one object have a distance.
  %one = arith.constant 1 : i32
  %record_integer = "udonhl.ptrtoint"(%record) : (!udonhl.ptr<0>) -> i32
  %head_integer = "udonhl.ptrtoint"(%head) : (!udonhl.ptr<0>) -> i32
  %null_integer = "udonhl.ptrtoint"(%null) : (!udonhl.ptr<0>) -> i32
  %not_null = arith.cmpi ne, %record_integer, %null_integer : i32
  %loaded_integer = "udonhl.ptrtoint"(%loaded) : (!udonhl.ptr<0>) -> i32
  %same = arith.cmpi eq, %loaded_integer, %record_integer : i32
  %different = arith.cmpi ne, %record_integer, %head_integer : i32
  %has_token = arith.cmpi ne, %token, %zero : i32
  %low_bits = arith.andi %head_integer, %one : i32
  %aligned = arith.cmpi eq, %low_bits, %zero : i32
  %distance = arith.subi %integer, %record_integer : i32
  %is_eight = arith.cmpi eq, %distance, %eight : i32
  %a = arith.andi %not_null, %same : i1
  %b = arith.andi %a, %different : i1
  %c = arith.andi %b, %has_token : i1
  %d = arith.andi %c, %aligned : i1
  %all = arith.andi %d, %is_eight : i1
  %flags = arith.constant 13 : i32
  %recorded = arith.select %all, %flags, %zero : i32
  %twelve = arith.constant 12 : i32
  %flags_address = "udonhl.ptr_add"(%record, %twelve) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
  "udonhl.store"(%recorded, %flags_address) <{alignment = 4 : i64}> : (i32, !udonhl.ptr<0>) -> ()

  %cleared = udonhl.address_of @cleared : !udonhl.ptr<0>
  "udonhl.store"(%null, %cleared) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
  %seven = arith.constant 7 : i32
  %tagged = "udonhl.inttoptr"(%seven) : (i32) -> !udonhl.ptr<0>
  %tag = udonhl.address_of @tag : !udonhl.ptr<0>
  "udonhl.store"(%tagged, %tag) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
  return %zero : i32
}
