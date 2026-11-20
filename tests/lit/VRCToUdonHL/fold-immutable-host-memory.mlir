// RUN: %vrc-opt --split-input-file --vrc-fold-immutable-host-memory %s | %FileCheck %s

// A raw global is closed-world state: a load from bytes no operation of the
// module can store to reads the initializer.

// The module reports what one run folded; a global nothing stores to is
// marked constant.
// CHECK: vrc.immutable_host_memory = {constant_globals = 1 : i64, folded_loads = 1 : i64}
// CHECK: alignment = 4 : i64, constant, initial_value = 42 : i32
// CHECK-SAME: sym_name = "limit"
"udonhl.global"() <{sym_name = "limit", storage_type = i32, initial_value = 42 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> : () -> ()
// CHECK: alignment = 4 : i64, initial_value = 0 : i32
// CHECK-SAME: sym_name = "counter"
"udonhl.global"() <{sym_name = "counter", storage_type = i32, initial_value = 0 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> : () -> ()

// CHECK-LABEL: func.func @never_stored() -> i32
// CHECK: %[[LIMIT:.*]] = arith.constant 42 : i32
// CHECK: %[[COUNT:.*]] = "udonhl.load"
// CHECK-NOT: udonhl.load
// CHECK: arith.addi %[[LIMIT]], %[[COUNT]]
func.func @never_stored() -> i32 {
  %limit_address = udonhl.address_of @limit : !udonhl.ptr<0>
  %limit = "udonhl.load"(%limit_address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> i32
  %counter_address = udonhl.address_of @counter : !udonhl.ptr<0>
  %count = "udonhl.load"(%counter_address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> i32
  %sum = arith.addi %limit, %count : i32
  "udonhl.store"(%sum, %counter_address) <{alignment = 4 : i64}> : (i32, !udonhl.ptr<0>) -> ()
  return %sum : i32
}

// -----

// Field sensitivity: a record that mixes a field written on every call with
// fields nothing writes. The store goes through an address passed to a
// private function, whose call sites are all visible.
"udonhl.global"() <{sym_name = "session", storage_type = !llvm.array<12 x i8>, initial_value = array<i8: 7, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0>, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> : () -> ()

func.func private @touch(%record: !udonhl.ptr<0>) {
  %four = arith.constant 4 : i32
  %state = "udonhl.ptr_add"(%record, %four) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
  %busy = arith.constant 2 : i32
  "udonhl.store"(%busy, %state) <{alignment = 4 : i64}> : (i32, !udonhl.ptr<0>) -> ()
  return
}

// CHECK-LABEL: func.func @fields() -> i32
// CHECK: %[[MAGIC:.*]] = arith.constant 7 : i32
// CHECK: %[[STATE:.*]] = "udonhl.load"
// CHECK-NOT: udonhl.load
// CHECK: %[[FLAG:.*]] = arith.constant true
// CHECK: arith.addi %[[MAGIC]], %[[STATE]]
// CHECK: arith.select %[[FLAG]]
func.func @fields() -> i32 {
  %session = udonhl.address_of @session : !udonhl.ptr<0>
  call @touch(%session) : (!udonhl.ptr<0>) -> ()
  %magic = "udonhl.load"(%session) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> i32
  %four = arith.constant 4 : i32
  %state_address = "udonhl.ptr_add"(%session, %four) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
  %state = "udonhl.load"(%state_address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> i32
  %nine = arith.constant 9 : i32
  %flag_address = "udonhl.ptr_add"(%session, %nine) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
  %flag = "udonhl.load"(%flag_address) <{alignment = 1 : i64}> : (!udonhl.ptr<0>) -> i1
  %sum = arith.addi %magic, %state : i32
  %zero = arith.constant 0 : i32
  %result = arith.select %flag, %sum, %zero : i32
  return %result : i32
}

// -----

// An array of 8-byte records {id, refs}. A store to `refs` of a run-time
// element writes the bytes 4..7 of every record, and no `id`.
"udonhl.global"() <{sym_name = "records", storage_type = !llvm.array<16 x i8>, initial_value = array<i8: 11, 0, 0, 0, 0, 0, 0, 0, 22, 0, 0, 0, 0, 0, 0, 0>, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> : () -> ()
// A store at a run-time byte offset may land anywhere in the object.
"udonhl.global"() <{sym_name = "bytes", storage_type = !llvm.array<8 x i8>, initial_value = array<i8: 1, 2, 3, 4, 5, 6, 7, 8>, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> : () -> ()

// CHECK-LABEL: func.func @strided(%{{.*}}: i32) -> i32
// CHECK: %[[ID:.*]] = arith.constant 22 : i32
// CHECK: %[[REFS:.*]] = "udonhl.load"
// CHECK: %[[BYTE:.*]] = "udonhl.load"
// CHECK: arith.addi %[[ID]], %[[REFS]]
func.func @strided(%index: i32) -> i32 {
  %records = udonhl.address_of @records : !udonhl.ptr<0>
  %eight = arith.constant 8 : i32
  %four = arith.constant 4 : i32
  %scaled = arith.muli %index, %eight : i32
  %record = "udonhl.ptr_add"(%records, %scaled) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
  %refs_address = "udonhl.ptr_add"(%record, %four) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
  %one = arith.constant 1 : i32
  "udonhl.store"(%one, %refs_address) <{alignment = 4 : i64}> : (i32, !udonhl.ptr<0>) -> ()

  %second = "udonhl.ptr_add"(%records, %eight) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
  %id = "udonhl.load"(%second) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> i32
  %twelve = arith.constant 12 : i32
  %second_refs = "udonhl.ptr_add"(%records, %twelve) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
  %refs = "udonhl.load"(%second_refs) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> i32

  %bytes = udonhl.address_of @bytes : !udonhl.ptr<0>
  %anywhere = "udonhl.ptr_add"(%bytes, %index) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
  %zero = arith.constant 0 : i8
  "udonhl.store"(%zero, %anywhere) <{alignment = 1 : i64}> : (i8, !udonhl.ptr<0>) -> ()
  %byte = "udonhl.load"(%bytes) <{alignment = 1 : i64}> : (!udonhl.ptr<0>) -> i8
  %wide = arith.extui %byte : i8 to i32
  %sum = arith.addi %id, %refs : i32
  %result = arith.addi %sum, %wide : i32
  return %result : i32
}

// -----

// A relocated word folds to the address it names and a token word to its
// Unity buffer, so the access behind the handle folds in the same run.
"udonhl.global"() <{sym_name = "value", storage_type = i32, initial_value = 5 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> : () -> ()
"udonhl.global"() <{sym_name = "table", storage_type = !llvm.array<8 x i8>, initial_value = array<i8: 0, 0, 0, 0, 0, 0, 0, 0>, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64,
  relocations = [{offset = 0 : i32, symbol = @value, addend = 0 : i32},
                 {offset = 4 : i32, assigned_buffer = {resource_id = 1 : i32, texture_id = 3 : i32, capacity_bytes = 64 : i64, format = "rgba8_unorm"}}]}> : () -> ()

// CHECK-LABEL: func.func @handles() -> i32
// CHECK-NOT: udonhl.load
// CHECK-DAG: %[[FIVE:.*]] = arith.constant 5 : i32
// CHECK-DAG: %[[BUFFER:.*]] = "vrc.assigned_buffer"() <{capacity_bytes = 64 : i64, format = "rgba8_unorm", resource_id = 1 : i32, texture_id = 3 : i32}>
// CHECK-DAG: %[[TOKEN:.*]] = builtin.unrealized_conversion_cast %[[BUFFER]] : !vrc.buffer to i32
// CHECK: arith.addi %[[FIVE]], %[[TOKEN]]
// CHECK: return
func.func @handles() -> i32 {
  %table = udonhl.address_of @table : !udonhl.ptr<0>
  %pointer = "udonhl.load"(%table) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
  %loaded = "udonhl.load"(%pointer) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> i32
  %four = arith.constant 4 : i32
  %token_address = "udonhl.ptr_add"(%table, %four) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
  %token = "udonhl.load"(%token_address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> i32
  %sum = arith.addi %loaded, %token : i32
  return %sum : i32
}

// -----

// The generated binding writes the bytes of a completed readback into the
// raw heap: the destination of a readback is stored to. An address handed to
// a function without a definition escapes, and a store through an address
// the analysis lost clobbers the escaped object, and only it.
"udonhl.global"() <{sym_name = "result", storage_type = i32, initial_value = 3 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> : () -> ()
"udonhl.global"() <{sym_name = "escaped", storage_type = i32, initial_value = 4 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> : () -> ()
"udonhl.global"() <{sym_name = "private", storage_type = i32, initial_value = 6 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> : () -> ()

func.func private @external(!udonhl.ptr<0>) -> !udonhl.ptr<0>

// CHECK-LABEL: func.func @writers(%{{.*}}: !vrc.buffer) -> i32
// CHECK: %[[RESULT:.*]] = "udonhl.load"
// CHECK: %[[ESCAPED:.*]] = "udonhl.load"
// CHECK-NOT: udonhl.load
// CHECK: %[[PRIVATE:.*]] = arith.constant 6 : i32
// CHECK: %[[SUM:.*]] = arith.addi %[[RESULT]], %[[ESCAPED]]
// CHECK: arith.addi %[[SUM]], %[[PRIVATE]]
func.func @writers(%buffer: !vrc.buffer) -> i32 attributes {vrc.async_effect = "may_suspend"} {
  %zero = arith.constant 0 : i32
  %four = arith.constant 4 : i32
  %result_address = udonhl.address_of @result : !udonhl.ptr<0>
  %destination = "udonhl.ptrtoint"(%result_address) : (!udonhl.ptr<0>) -> i32
  %event = vrc.buffer_readback %buffer, %zero, %four, %destination : (!vrc.buffer, i32, i32, i32) -> !vrc.event
  %status = vrc.await %event : (!vrc.event) -> i32
  %result = "udonhl.load"(%result_address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> i32

  %escaped_address = udonhl.address_of @escaped : !udonhl.ptr<0>
  %unknown = call @external(%escaped_address) : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
  "udonhl.store"(%zero, %unknown) <{alignment = 4 : i64}> : (i32, !udonhl.ptr<0>) -> ()
  %escaped = "udonhl.load"(%escaped_address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> i32
  %private_address = udonhl.address_of @private : !udonhl.ptr<0>
  %private = "udonhl.load"(%private_address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> i32

  %sum = arith.addi %result, %escaped : i32
  %total = arith.addi %sum, %private : i32
  return %total : i32
}
