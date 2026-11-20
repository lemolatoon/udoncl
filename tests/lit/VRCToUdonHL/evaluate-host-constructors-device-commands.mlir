// RUN: %vrc-opt --split-input-file --vrc-evaluate-host-constructors --symbol-dce %s | %FileCheck %s

// Device state is not Host memory, so a Device command a static constructor
// issues is not executed at build time: it stays behind, in issue order and
// with constant operands, as the residual body of the constructor.

// The handle table the constructor fills in; the commands read their buffers
// back from it.
// CHECK: relocations = [{assigned_buffer = {capacity_bytes = 64 : i64, format = "rgba8_unorm", resource_id = 0 : i32, texture_id = 3 : i32}, offset = 0 : i32}, {assigned_buffer = {capacity_bytes = 64 : i64, format = "rgba8_unorm", resource_id = 1 : i32, texture_id = 4 : i32}, offset = 4 : i32}]
// CHECK-SAME: sym_name = "handles"
"udonhl.global"() <{sym_name = "handles", storage_type = !llvm.array<8 x i8>, initial_value = array<i8: 0, 0, 0, 0, 0, 0, 0, 0>, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> : () -> ()
// The pattern cell is overwritten after the first fill; the image keeps the
// last value, and each fill keeps the pattern it read when it was issued.
// CHECK: initial_value = 9 : i32
// CHECK-SAME: sym_name = "pattern"
"udonhl.global"() <{sym_name = "pattern", storage_type = i32, initial_value = 0 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> : () -> ()
// CHECK: "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, constant, initial_value = array<i8: 0, 0, 0, 0>, linkage = "internal", storage_type = !llvm.array<4 x i8>, sym_name = "__udoncl_static_fill_pattern_0"}>
// CHECK: "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, constant, initial_value = array<i8: 9, 0, 0, 0>, linkage = "internal", storage_type = !llvm.array<4 x i8>, sym_name = "__udoncl_static_fill_pattern_1"}>

func.func private @clear(%handle: !udonhl.ptr<0>, %bytes: i32) -> i32 attributes {vrc.async_effect = "may_suspend"} {
  %zero = arith.constant 0 : i32
  %four = arith.constant 4 : i32
  %token = "udonhl.load"(%handle) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> i32
  %buffer = builtin.unrealized_conversion_cast %token : i32 to !vrc.buffer
  %pattern = udonhl.address_of @pattern : !udonhl.ptr<0>
  %pointer = "udonhl.ptrtoint"(%pattern) : (!udonhl.ptr<0>) -> i32
  %event = vrc.buffer_fill %buffer, %zero, %bytes, %pointer, %four : (!vrc.buffer, i32, i32, i32, i32) -> !vrc.event
  %status = vrc.await %event : (!vrc.event) -> i32
  return %status : i32
}

// CHECK-LABEL: func.func @prime() -> i32
// CHECK-SAME: vrc.async_effect = "may_suspend"
// CHECK-SAME: residual_commands = 4 : i32

// Fill buffer 3 with the zero word.
// CHECK: %[[FIRST:.*]] = "vrc.assigned_buffer"() <{capacity_bytes = 64 : i64, format = "rgba8_unorm", resource_id = 0 : i32, texture_id = 3 : i32}>
// CHECK: %[[ZERO_PATTERN:.*]] = udonhl.address_of @__udoncl_static_fill_pattern_0
// CHECK: %[[ZERO_POINTER:.*]] = "udonhl.ptrtoint"(%[[ZERO_PATTERN]])
// CHECK: %[[FILL0:.*]] = vrc.buffer_fill %[[FIRST]], %{{.*}}, %{{.*}}, %[[ZERO_POINTER]], %{{.*}} :
// Every command is awaited: the Device work scheduler turns each await into
// a slice budget checkpoint.
// CHECK-NEXT: vrc.await %[[FILL0]]

// Fill buffer 4 with the pattern the cell held then, 9.
// CHECK: %[[SECOND:.*]] = "vrc.assigned_buffer"() <{capacity_bytes = 64 : i64, format = "rgba8_unorm", resource_id = 1 : i32, texture_id = 4 : i32}>
// CHECK: %[[NINE_PATTERN:.*]] = udonhl.address_of @__udoncl_static_fill_pattern_1
// CHECK: %[[NINE_POINTER:.*]] = "udonhl.ptrtoint"(%[[NINE_PATTERN]])
// CHECK: %[[FILL1:.*]] = vrc.buffer_fill %[[SECOND]], %{{.*}}, %{{.*}}, %[[NINE_POINTER]], %{{.*}} :
// CHECK-NEXT: vrc.await %[[FILL1]]

// Copy 32 bytes from buffer 3 to byte 16 of buffer 4.
// CHECK: %[[SOURCE:.*]] = "vrc.assigned_buffer"() <{{.*}}texture_id = 3 : i32}>
// CHECK: %[[DESTINATION:.*]] = "vrc.assigned_buffer"() <{{.*}}texture_id = 4 : i32}>
// CHECK: %[[C0:.*]] = arith.constant 0 : i32
// CHECK: %[[C16:.*]] = arith.constant 16 : i32
// CHECK: %[[C32:.*]] = arith.constant 32 : i32
// CHECK: %[[COPY:.*]] = vrc.buffer_copy %[[SOURCE]], %[[DESTINATION]], %[[C0]], %[[C16]], %[[C32]] :
// CHECK-NEXT: vrc.await %[[COPY]]

// Dispatch kernel 7 from a static tensor and buffer 3 into buffer 4 with the
// copied scalars 5 and 0.25. The descriptor directly precedes its dispatch
// and every attribute of the site is kept.
// CHECK: %[[TENSOR:.*]] = "vrc.static_buffer"() <{byte_size = 16 : i32, resource_id = 2 : i32}>
// CHECK: %[[INPUT:.*]] = "vrc.assigned_buffer"() <{{.*}}texture_id = 3 : i32}>
// CHECK: %[[OUTPUT:.*]] = "vrc.assigned_buffer"() <{{.*}}texture_id = 4 : i32}>
// CHECK: %[[COUNT:.*]] = arith.constant 5 : i32
// CHECK: %[[SCALE:.*]] = arith.constant 2.500000e-01 : f32
// CHECK-NEXT: %[[DESCRIPTOR:.*]] = vrc.dispatch_desc_create %[[COUNT]], %[[SCALE]] {kernel_id = 7 : i32, scalar_kinds = ["i32", "f32"]}
// CHECK-NEXT: %[[DISPATCH:.*]] = vrc.dispatch %[[DESCRIPTOR]](%[[TENSOR]], %[[INPUT]], %[[OUTPUT]]) {argument_kinds = ["buffer", "buffer", "buffer", "i32", "f32"], global_offset = array<i32: 0, 0, 0>, global_size = array<i32: 8, 1, 1>, kernel_id = 7 : i32, local_size = array<i32: 1, 1, 1>, resource_accesses = ["read", "read", "write"], vrc.site = "prime"}
// CHECK-NEXT: vrc.await %[[DISPATCH]]
// CHECK: %[[STATUS:.*]] = arith.constant 0 : i32
// CHECK-NEXT: return %[[STATUS]] : i32
// CHECK-NEXT: }

// The helper only ran at build time.
// CHECK-NOT: @clear
func.func @prime() -> i32 attributes {vrc.static_constructor, vrc.async_effect = "may_suspend"} {
  %zero = arith.constant 0 : i32
  %four = arith.constant 4 : i32
  %handles = udonhl.address_of @handles : !udonhl.ptr<0>
  %first = "vrc.assigned_buffer"() <{capacity_bytes = 64 : i64, format = "rgba8_unorm", resource_id = 0 : i32, texture_id = 3 : i32}> : () -> !vrc.buffer
  %first_token = builtin.unrealized_conversion_cast %first : !vrc.buffer to i32
  "udonhl.store"(%first_token, %handles) <{alignment = 4 : i64}> : (i32, !udonhl.ptr<0>) -> ()
  %second = "vrc.assigned_buffer"() <{capacity_bytes = 64 : i64, format = "rgba8_unorm", resource_id = 1 : i32, texture_id = 4 : i32}> : () -> !vrc.buffer
  %second_token = builtin.unrealized_conversion_cast %second : !vrc.buffer to i32
  %second_handle = "udonhl.ptr_add"(%handles, %four) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
  "udonhl.store"(%second_token, %second_handle) <{alignment = 4 : i64}> : (i32, !udonhl.ptr<0>) -> ()

  %bytes = arith.constant 64 : i32
  %cleared = call @clear(%handles, %bytes) : (!udonhl.ptr<0>, i32) -> i32
  %pattern = udonhl.address_of @pattern : !udonhl.ptr<0>
  %nine = arith.constant 9 : i32
  "udonhl.store"(%nine, %pattern) <{alignment = 4 : i64}> : (i32, !udonhl.ptr<0>) -> ()
  %filled = call @clear(%second_handle, %bytes) : (!udonhl.ptr<0>, i32) -> i32

  %sixteen = arith.constant 16 : i32
  %half = arith.constant 32 : i32
  %copy = vrc.buffer_copy %first, %second, %zero, %sixteen, %half : (!vrc.buffer, !vrc.buffer, i32, i32, i32) -> !vrc.event
  %copied = vrc.await %copy : (!vrc.event) -> i32

  %tensor = "vrc.static_buffer"() <{byte_size = 16 : i32, resource_id = 2 : i32}> : () -> !vrc.buffer
  %one = arith.constant 1 : i32
  %count = arith.addi %four, %one : i32
  %scale = arith.constant 0.25 : f32
  %descriptor = vrc.dispatch_desc_create %count, %scale {kernel_id = 7 : i32, scalar_kinds = ["i32", "f32"]} : (i32, f32) -> !vrc.dispatch_desc
  %dispatch = vrc.dispatch %descriptor(%tensor, %first, %second) {argument_kinds = ["buffer", "buffer", "buffer", "i32", "f32"], global_offset = array<i32: 0, 0, 0>, global_size = array<i32: 8, 1, 1>, kernel_id = 7 : i32, local_size = array<i32: 1, 1, 1>, resource_accesses = ["read", "read", "write"], vrc.site = "prime"} : (!vrc.dispatch_desc, !vrc.buffer, !vrc.buffer, !vrc.buffer) -> !vrc.event
  %dispatched = vrc.await %dispatch : (!vrc.event) -> i32

  %a = arith.ori %cleared, %filled : i32
  %b = arith.ori %a, %copied : i32
  %status = arith.ori %b, %dispatched : i32
  return %status : i32
}

// -----

// A constructor that could never suspend stays synchronous: its commands
// are issued without checkpoints.
"udonhl.global"() <{sym_name = "word", storage_type = i32, initial_value = 7 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> : () -> ()

// CHECK-LABEL: func.func @synchronous()
// CHECK-NOT: vrc.async_effect
// CHECK-SAME: residual_commands = 1 : i32
// CHECK: vrc.buffer_fill
// CHECK-NOT: vrc.await
// CHECK: return
func.func @synchronous() attributes {vrc.static_constructor} {
  %zero = arith.constant 0 : i32
  %four = arith.constant 4 : i32
  %buffer = "vrc.assigned_buffer"() <{capacity_bytes = 64 : i64, format = "rgba8_unorm", resource_id = 0 : i32, texture_id = 3 : i32}> : () -> !vrc.buffer
  %word = udonhl.address_of @word : !udonhl.ptr<0>
  %pointer = "udonhl.ptrtoint"(%word) : (!udonhl.ptr<0>) -> i32
  %event = vrc.buffer_fill %buffer, %zero, %four, %pointer, %four : (!vrc.buffer, i32, i32, i32, i32) -> !vrc.event
  return
}

// -----

// A constructor that could suspend but leaves no command still completes
// asynchronously, because that is the export's public ABI: it yields once.
"udonhl.global"() <{sym_name = "ready", storage_type = i32, initial_value = 0 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> : () -> ()

func.func private @wait_if_pending(%pending: i32) -> i32 attributes {vrc.async_effect = "may_suspend"} {
  %zero = arith.constant 0 : i32
  %complete = arith.cmpi eq, %pending, %zero : i32
  cf.cond_br %complete, ^done(%zero : i32), ^wait
^wait:
  %event = builtin.unrealized_conversion_cast %pending : i32 to !vrc.event
  %status = vrc.await %event : (!vrc.event) -> i32
  cf.br ^done(%status : i32)
^done(%result: i32):
  return %result : i32
}

// CHECK-LABEL: func.func @asynchronous() -> i32
// CHECK-SAME: vrc.async_effect = "may_suspend"
// CHECK-SAME: residual_commands = 0 : i32
// CHECK: %[[COMPLETE:.*]] = arith.constant 0 : i32
// CHECK: %[[EVENT:.*]] = builtin.unrealized_conversion_cast %[[COMPLETE]] : i32 to !vrc.event
// CHECK: vrc.await %[[EVENT]] {vrc.cooperative_checkpoint}
// CHECK: return
func.func @asynchronous() -> i32 attributes {vrc.static_constructor} {
  %zero = arith.constant 0 : i32
  %status = call @wait_if_pending(%zero) : (i32) -> i32
  %one = arith.constant 1 : i32
  %ready = udonhl.address_of @ready : !udonhl.ptr<0>
  "udonhl.store"(%one, %ready) <{alignment = 4 : i64}> : (i32, !udonhl.ptr<0>) -> ()
  return %status : i32
}
