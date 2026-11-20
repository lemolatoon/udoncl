// RUN: %vrc-opt --split-input-file --verify-diagnostics --vrc-evaluate-host-constructors %s

// A static constructor that cannot be evaluated is a compile error at the
// blocking operation. Nothing falls back to run-time initialization.

func.func private @external() -> i32

// expected-note @below {{while evaluating @calls_declaration}}
func.func @calls_declaration() -> i32 attributes {vrc.static_constructor} {
  // expected-error @below {{cannot be evaluated at build time: calls @external, which has no definition}}
  %status = call @external() : () -> i32
  return %status : i32
}

// -----

"udonhl.global"() <{sym_name = "destination", storage_type = i32, initial_value = 0 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> : () -> ()

// A readback observes Device state, which no build knows.
// expected-note @below {{while evaluating @reads_back}}
func.func @reads_back() -> i32 attributes {vrc.static_constructor, vrc.async_effect = "may_suspend"} {
  %buffer = "vrc.assigned_buffer"() <{capacity_bytes = 64 : i64, format = "rgba8_unorm", resource_id = 0 : i32, texture_id = 1 : i32}> : () -> !vrc.buffer
  %zero = arith.constant 0 : i32
  %four = arith.constant 4 : i32
  %address = udonhl.address_of @destination : !udonhl.ptr<0>
  %pointer = "udonhl.ptrtoint"(%address) : (!udonhl.ptr<0>) -> i32
  // expected-error @below {{cannot be evaluated at build time: is not a function of the build}}
  %event = vrc.buffer_readback %buffer, %zero, %four, %pointer : (!vrc.buffer, i32, i32, i32) -> !vrc.event
  %status = vrc.await %event : (!vrc.event) -> i32
  return %status : i32
}

// -----

// Only the event of a command that is complete at issue is known.
// expected-note @below {{while evaluating @waits}}
func.func @waits() -> i32 attributes {vrc.static_constructor, vrc.async_effect = "may_suspend"} {
  %pending = arith.constant 3 : i32
  %event = builtin.unrealized_conversion_cast %pending : i32 to !vrc.event
  // expected-error @below {{cannot be evaluated at build time: waits for a Device event that is not complete at issue}}
  %status = vrc.await %event : (!vrc.event) -> i32
  return %status : i32
}

// -----

// The diagnostic carries the call path.
// expected-note @below {{while evaluating @helper}}
func.func private @helper(%pointer: !udonhl.ptr<0>) -> i32 {
  // expected-error @below {{cannot be evaluated at build time: accesses the absolute address 0}}
  %value = "udonhl.load"(%pointer) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> i32
  return %value : i32
}

// expected-note @below {{while evaluating @dereferences_null}}
func.func @dereferences_null() -> i32 attributes {vrc.static_constructor} {
  %zero = arith.constant 0 : i32
  %null = "udonhl.inttoptr"(%zero) : (i32) -> !udonhl.ptr<0>
  %value = call @helper(%null) : (!udonhl.ptr<0>) -> i32
  return %value : i32
}

// -----

"udonhl.global"() <{sym_name = "word", storage_type = i32, initial_value = 0 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> : () -> ()

// expected-note @below {{while evaluating @overruns}}
func.func @overruns() attributes {vrc.static_constructor} {
  %address = udonhl.address_of @word : !udonhl.ptr<0>
  %two = arith.constant 2 : i32
  %inside = "udonhl.ptr_add"(%address, %two) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
  %value = arith.constant 1 : i32
  // expected-error @below {{cannot be evaluated at build time: accesses memory outside its object}}
  "udonhl.store"(%value, %inside) <{alignment = 1 : i64}> : (i32, !udonhl.ptr<0>) -> ()
  return
}

// -----

"udonhl.global"() <{sym_name = "table", storage_type = i32, initial_value = 3 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64, constant}> : () -> ()

// expected-note @below {{while evaluating @writes_constant}}
func.func @writes_constant() attributes {vrc.static_constructor} {
  %address = udonhl.address_of @table : !udonhl.ptr<0>
  %value = arith.constant 1 : i32
  // expected-error @below {{cannot be evaluated at build time: stores to constant memory}}
  "udonhl.store"(%value, %address) <{alignment = 4 : i64}> : (i32, !udonhl.ptr<0>) -> ()
  return
}

// -----

"udonhl.global"() <{sym_name = "cell", storage_type = !udonhl.ptr<0>, initial_value = 0 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> : () -> ()

// The bytes of an address do not exist before the heap is laid out.
// expected-note @below {{while evaluating @splits_address}}
func.func @splits_address() attributes {vrc.static_constructor} {
  %address = udonhl.address_of @cell : !udonhl.ptr<0>
  "udonhl.store"(%address, %address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
  %byte = arith.constant 1 : i8
  // expected-error @below {{cannot be evaluated at build time: partially overwrites a stored pointer or buffer token}}
  "udonhl.store"(%byte, %address) <{alignment = 1 : i64}> : (i8, !udonhl.ptr<0>) -> ()
  return
}

// -----

"udonhl.global"() <{sym_name = "first", storage_type = i32, initial_value = 0 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> : () -> ()
"udonhl.global"() <{sym_name = "second", storage_type = i32, initial_value = 0 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> : () -> ()

// The order of two objects is a property of the layout.
// expected-note @below {{while evaluating @orders_objects}}
func.func @orders_objects() -> i32 attributes {vrc.static_constructor} {
  %first = udonhl.address_of @first : !udonhl.ptr<0>
  %second = udonhl.address_of @second : !udonhl.ptr<0>
  %left = "udonhl.ptrtoint"(%first) : (!udonhl.ptr<0>) -> i32
  %right = "udonhl.ptrtoint"(%second) : (!udonhl.ptr<0>) -> i32
  // expected-error @below {{cannot be evaluated at build time: compares addresses of different objects}}
  %before = arith.cmpi ult, %left, %right : i32
  %status = arith.extui %before : i1 to i32
  return %status : i32
}

// -----

"udonhl.global"() <{sym_name = "object", storage_type = i32, initial_value = 0 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> : () -> ()

// expected-note @below {{while evaluating @hashes_address}}
func.func @hashes_address() -> i32 attributes {vrc.static_constructor} {
  %address = udonhl.address_of @object : !udonhl.ptr<0>
  %integer = "udonhl.ptrtoint"(%address) : (!udonhl.ptr<0>) -> i32
  %three = arith.constant 3 : i32
  // expected-error @below {{cannot be evaluated at build time: does unsupported arithmetic on an address}}
  %hash = arith.muli %integer, %three : i32
  return %hash : i32
}

// -----

// expected-note @below {{while evaluating @divides}}
func.func @divides() -> i32 attributes {vrc.static_constructor} {
  %one = arith.constant 1 : i32
  %zero = arith.constant 0 : i32
  // expected-error @below {{cannot be evaluated at build time: divides by zero or overflows}}
  %quotient = arith.divsi %one, %zero : i32
  return %quotient : i32
}

// -----

// A recursive run is named once.
// expected-note @below {{while evaluating @recurses}}
func.func private @recurses() -> i32 {
  // expected-error @below {{cannot be evaluated at build time: exceeds the call depth limit}}
  %value = call @recurses() : () -> i32
  return %value : i32
}

// expected-note @below {{while evaluating @overflows_stack}}
func.func @overflows_stack() -> i32 attributes {vrc.static_constructor} {
  %value = call @recurses() : () -> i32
  return %value : i32
}

// -----

// The validation that used to run at start-up fails the build instead.
// expected-error @below {{static constructor fails at build time with status 5}}
func.func @fails() -> i32 attributes {vrc.static_constructor} {
  %status = arith.constant 5 : i32
  return %status : i32
}

// -----

"udonhl.global"() <{sym_name = "escaped", storage_type = !udonhl.ptr<0>, initial_value = 0 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> : () -> ()

// Automatic storage has no address an initializer could name.
// expected-error @below {{static constructor leaves the address of automatic or array storage in @escaped at offset 0}}
func.func @leaks_automatic_storage() attributes {vrc.static_constructor} {
  %four = arith.constant 4 : i32
  %local = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
  %cell = udonhl.address_of @escaped : !udonhl.ptr<0>
  "udonhl.store"(%local, %cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
  return
}

// -----

// expected-error @below {{static constructor must be a definition without parameters that returns void or a 32-bit status}}
func.func @takes_parameter(%value: i32) -> i32 attributes {vrc.static_constructor} {
  return %value : i32
}
