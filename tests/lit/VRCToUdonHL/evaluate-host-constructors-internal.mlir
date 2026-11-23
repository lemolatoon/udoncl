// RUN: %vrc-opt --split-input-file --verify-diagnostics --vrc-evaluate-host-constructors %s | %FileCheck %s

// A C constructor that is not also an export was configured as one only so
// the Host analyses saw it as a root. Once it has run it leaves the exports.

"udonhl.global"() <{sym_name = "ready", storage_type = i32, initial_value = 0 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> : () -> ()

// CHECK: initial_value = 1 : i32
// CHECK-SAME: sym_name = "ready"
// CHECK-LABEL: func.func private @initialize()
// CHECK-SAME: vrc.evaluated_static_constructor
// CHECK-NOT: vrc.export_id
// CHECK-NOT: vrc.internal_static_constructor
// CHECK-NOT: vrc.public_name
func.func @initialize() attributes {vrc.static_constructor, vrc.internal_static_constructor, vrc.export_id = 5 : i32, vrc.public_name = "Initialize", vrc.parameter_names = []} {
  %one = arith.constant 1 : i32
  %ready = udonhl.address_of @ready : !udonhl.ptr<0>
  "udonhl.store"(%one, %ready) <{alignment = 4 : i64}> : (i32, !udonhl.ptr<0>) -> ()
  return
}

// -----

"udonhl.global"() <{sym_name = "pattern", storage_type = i32, initial_value = 0 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> : () -> ()

// A Device command it leaves would need U# to run it.
// expected-error@below {{the constructor leaves 1 Device commands to run; mark it UDONCL_EXPORT so U# runs them}}
func.func @prime() -> i32 attributes {vrc.static_constructor, vrc.internal_static_constructor, vrc.async_effect = "may_suspend", vrc.export_id = 5 : i32} {
  %zero = arith.constant 0 : i32
  %four = arith.constant 4 : i32
  %bytes = arith.constant 64 : i32
  %buffer = "vrc.assigned_buffer"() <{capacity_bytes = 64 : i64, format = "rgba8_unorm", resource_id = 0 : i32, texture_id = 3 : i32}> : () -> !vrc.buffer
  %pattern = udonhl.address_of @pattern : !udonhl.ptr<0>
  %pointer = "udonhl.ptrtoint"(%pattern) : (!udonhl.ptr<0>) -> i32
  %event = vrc.buffer_fill %buffer, %zero, %bytes, %pointer, %four : (!vrc.buffer, i32, i32, i32, i32) -> !vrc.event
  %status = vrc.await %event : (!vrc.event) -> i32
  return %status : i32
}
