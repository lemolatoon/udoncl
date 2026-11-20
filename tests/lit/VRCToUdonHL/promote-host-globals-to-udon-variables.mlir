// RUN: %vrc-opt --vrc-promote-host-globals-to-udon-variables %s | %FileCheck %s

// A scalar global whose address only feeds whole-object loads and stores of
// its own type has no observable address. It becomes a Udon variable: the
// accesses name the symbol and the address computations disappear.

// CHECK: "udonhl.global"() <{{.*}}sym_name = "counter"{{.*}}> {vrc.udon_variable}
"udonhl.global"() <{sym_name = "counter", storage_type = i32, initial_value = 7 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> : () -> ()
// CHECK: "udonhl.global"() <{{.*}}sym_name = "cursor"{{.*}}> {vrc.udon_variable}
"udonhl.global"() <{sym_name = "cursor", storage_type = !udonhl.ptr<0>, initial_value = 0 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> : () -> ()

// CHECK-LABEL: func.func @count
// CHECK-NOT: udonhl.address_of
// CHECK: %[[OLD:.*]] = udonhl.variable_load @counter : i32
// CHECK: %[[NEW:.*]] = arith.addi %[[OLD]], %{{.*}} : i32
// CHECK: udonhl.variable_store %[[NEW]], @counter : i32
// CHECK: %[[P:.*]] = udonhl.variable_load @cursor : !udonhl.ptr<0>
// CHECK: udonhl.variable_store %[[P]], @cursor : !udonhl.ptr<0>
func.func @count() -> i32 {
  %address = udonhl.address_of @counter : !udonhl.ptr<0>
  %zero = arith.constant 0 : i32
  %same = "udonhl.ptr_add"(%address, %zero) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
  %old = "udonhl.load"(%same) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> i32
  %one = arith.constant 1 : i32
  %new = arith.addi %old, %one : i32
  "udonhl.store"(%new, %address) <{alignment = 4 : i64}> : (i32, !udonhl.ptr<0>) -> ()
  %cursor = udonhl.address_of @cursor : !udonhl.ptr<0>
  %p = "udonhl.load"(%cursor) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
  "udonhl.store"(%p, %cursor) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
  return %new : i32
}

// Every other use of the address keeps the global in the raw heap.

// A run-time offset.
// CHECK: "udonhl.global"() <{{.*}}sym_name = "indexed"{{.*}}> : () -> ()
"udonhl.global"() <{sym_name = "indexed", storage_type = i32, initial_value = 0 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> : () -> ()
// The address passed to a call.
// CHECK: "udonhl.global"() <{{.*}}sym_name = "passed"{{.*}}> : () -> ()
"udonhl.global"() <{sym_name = "passed", storage_type = i32, initial_value = 0 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> : () -> ()
// The address stored as a value.
// CHECK: "udonhl.global"() <{{.*}}sym_name = "published"{{.*}}> : () -> ()
"udonhl.global"() <{sym_name = "published", storage_type = !udonhl.ptr<0>, initial_value = 0 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> : () -> ()
// A byte of the object read on its own.
// CHECK: "udonhl.global"() <{{.*}}sym_name = "bytes"{{.*}}> : () -> ()
"udonhl.global"() <{sym_name = "bytes", storage_type = i32, initial_value = 0 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> : () -> ()
// A word the raw heap image relocates, and the global whose address that
// word publishes.
// CHECK: "udonhl.global"() <{{.*}}sym_name = "relocated"{{.*}}> : () -> ()
"udonhl.global"() <{sym_name = "relocated", storage_type = !udonhl.ptr<0>, initial_value = 0 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64, relocations = [{offset = 0 : i32, symbol = @named, addend = 0 : i32}]}> : () -> ()
// CHECK: "udonhl.global"() <{{.*}}sym_name = "named"{{.*}}> : () -> ()
"udonhl.global"() <{sym_name = "named", storage_type = i32, initial_value = 3 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> : () -> ()
// A bulk memory operation.
// CHECK: "udonhl.global"() <{{.*}}sym_name = "cleared"{{.*}}> : () -> ()
"udonhl.global"() <{sym_name = "cleared", storage_type = i32, initial_value = 0 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> : () -> ()

// CHECK-LABEL: func.func @escapes
// CHECK: udonhl.address_of @indexed
// CHECK: udonhl.address_of @passed
// CHECK: udonhl.address_of @published
// CHECK: udonhl.address_of @bytes
// CHECK: udonhl.address_of @cleared
// CHECK-NOT: udonhl.variable_
func.func private @callee(!udonhl.ptr<0>)
func.func @escapes(%offset: i32) -> i32 {
  %indexed = udonhl.address_of @indexed : !udonhl.ptr<0>
  %element = "udonhl.ptr_add"(%indexed, %offset) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
  %value = "udonhl.load"(%element) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> i32
  %passed = udonhl.address_of @passed : !udonhl.ptr<0>
  call @callee(%passed) : (!udonhl.ptr<0>) -> ()
  %published = udonhl.address_of @published : !udonhl.ptr<0>
  "udonhl.store"(%published, %published) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
  %bytes = udonhl.address_of @bytes : !udonhl.ptr<0>
  %byte = "udonhl.load"(%bytes) <{alignment = 1 : i64}> : (!udonhl.ptr<0>) -> i8
  %cleared = udonhl.address_of @cleared : !udonhl.ptr<0>
  %zero = arith.constant 0 : i8
  %four = arith.constant 4 : i32
  "udonhl.memset"(%cleared, %zero, %four) : (!udonhl.ptr<0>, i8, i32) -> ()
  return %value : i32
}

// A type that does not fill exactly one 32-bit integer slot, and an
// initializer the Udon assembler cannot spell, stay in the raw heap too.

// CHECK: "udonhl.global"() <{{.*}}sym_name = "wide"{{.*}}> : () -> ()
"udonhl.global"() <{sym_name = "wide", storage_type = i64, initial_value = 0 : i64, linkage = "internal", address_space = 0 : i32, alignment = 8 : i64}> : () -> ()
// CHECK: "udonhl.global"() <{{.*}}sym_name = "minimum"{{.*}}> : () -> ()
"udonhl.global"() <{sym_name = "minimum", storage_type = i32, initial_value = -2147483648 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> : () -> ()

// CHECK-LABEL: func.func @unpromotable
// CHECK: udonhl.address_of @wide
// CHECK: udonhl.address_of @minimum
// CHECK-NOT: udonhl.variable_
func.func @unpromotable() -> i32 {
  %wide = udonhl.address_of @wide : !udonhl.ptr<0>
  %long = "udonhl.load"(%wide) <{alignment = 8 : i64}> : (!udonhl.ptr<0>) -> i64
  %minimum = udonhl.address_of @minimum : !udonhl.ptr<0>
  %value = "udonhl.load"(%minimum) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> i32
  return %value : i32
}
