// RUN: %vrc-opt --split-input-file --verify-diagnostics --vrc-promote-host-globals-to-udon-variables %s | %FileCheck %s

// A file-scope udon_string_t or udon_int32_array_t refers to a managed Udon
// object, which the byte memory of the raw heap cannot hold: written in one
// export and read in another, it is a Udon variable of its own type.

// CHECK: "udonhl.global"() <{{.*}}sym_name = "message"{{.*}}> {vrc.udon_variable}
// CHECK: "udonhl.global"() <{{.*}}sym_name = "destination"{{.*}}> {vrc.udon_variable}
// CHECK-LABEL: func.func @begin
// CHECK: udonhl.variable_store %{{.*}}, @message : !udonhl.ptr<200>
// CHECK-LABEL: func.func @resume
// CHECK: udonhl.variable_load @message : !udonhl.ptr<200>
// CHECK: udonhl.variable_load @destination : !udonhl.ptr<201>
"udonhl.global"() <{sym_name = "message", storage_type = !udonhl.ptr<200>, initial_value = 0 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> : () -> ()
"udonhl.global"() <{sym_name = "destination", storage_type = !udonhl.ptr<201>, initial_value = 0 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> : () -> ()
func.func @begin(%text: !udonhl.ptr<200>, %array: !udonhl.ptr<201>) {
  %message = udonhl.address_of @message : !udonhl.ptr<0>
  "udonhl.store"(%text, %message) <{alignment = 4 : i64}> : (!udonhl.ptr<200>, !udonhl.ptr<0>) -> ()
  %destination = udonhl.address_of @destination : !udonhl.ptr<0>
  "udonhl.store"(%array, %destination) <{alignment = 4 : i64}> : (!udonhl.ptr<201>, !udonhl.ptr<0>) -> ()
  return
}
func.func @resume() -> (!udonhl.ptr<200>, !udonhl.ptr<201>) {
  %message = udonhl.address_of @message : !udonhl.ptr<0>
  %text = "udonhl.load"(%message) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<200>
  %destination = udonhl.address_of @destination : !udonhl.ptr<0>
  %array = "udonhl.load"(%destination) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<201>
  return %text, %array : !udonhl.ptr<200>, !udonhl.ptr<201>
}

// -----

// A managed reference whose address is observable cannot be a Udon variable,
// and the raw heap cannot hold it: an error naming the variable.

// expected-error @below {{the Host variable message holds a udon_string_t but cannot be a Udon variable}}
"udonhl.global"() <{sym_name = "message", storage_type = !udonhl.ptr<200>, initial_value = 0 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> : () -> ()
func.func private @keep(!udonhl.ptr<0>)
func.func @escape() {
  %message = udonhl.address_of @message : !udonhl.ptr<0>
  func.call @keep(%message) : (!udonhl.ptr<0>) -> ()
  return
}
