// RUN: %vrc-opt --vrc-fold-constant-host-tables %s | %FileCheck %s

// Pointer words of constant tables read at constant offsets are their
// initializers: the global a relocation names (plus its addend), or the
// integer the bytes hold.
module {
  "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, constant, initial_value = array<i8: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>, linkage = "internal", relocations = [{addend = 0 : i32, offset = 0 : i32, symbol = @first}, {addend = 2 : i32, offset = 4 : i32, symbol = @second}], storage_type = !llvm.array<3 x ptr>, sym_name = "names"}> : () -> ()
  "udonhl.global"() <{address_space = 0 : i32, alignment = 1 : i64, constant, initial_value = array<i8: 97, 0>, linkage = "internal", storage_type = !llvm.array<2 x i8>, sym_name = "first"}> : () -> ()
  "udonhl.global"() <{address_space = 0 : i32, alignment = 1 : i64, constant, initial_value = array<i8: 120, 120, 98, 0>, linkage = "internal", storage_type = !llvm.array<4 x i8>, sym_name = "second"}> : () -> ()
  "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, initial_value = array<i8: 0, 0, 0, 0>, linkage = "internal", relocations = [{addend = 0 : i32, offset = 0 : i32, symbol = @first}], storage_type = !llvm.array<1 x ptr>, sym_name = "mutable_names"}> : () -> ()
  "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, constant, initial_value = array<i8: 0, 0, 0, 0>, linkage = "internal", relocations = [{assigned_buffer = {capacity_bytes = 4 : i64, format = "rgba8_unorm", resource_id = 0 : i32, texture_id = 1 : i32}, offset = 0 : i32}], storage_type = !llvm.array<1 x ptr>, sym_name = "tokens"}> : () -> ()
  func.func private @use(!udonhl.ptr<0>)
  // CHECK-LABEL: func.func @Initialize
  func.func @Initialize(%dynamic: i32) attributes {udoncl.export} {
    %c4 = arith.constant 4 : i32
    %c8 = arith.constant 8 : i32
    %names = udonhl.address_of @names : <0>
    // CHECK: %[[FIRST:.*]] = udonhl.address_of @first
    // CHECK-NEXT: call @use(%[[FIRST]])
    %first = "udonhl.load"(%names) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    func.call @use(%first) : (!udonhl.ptr<0>) -> ()
    // CHECK: %[[SECOND:.*]] = udonhl.address_of @second
    // CHECK-NEXT: %[[ADDEND:.*]] = arith.constant 2 : i32
    // CHECK-NEXT: %[[MOVED:.*]] = "udonhl.ptr_add"(%[[SECOND]], %[[ADDEND]])
    // CHECK-NEXT: call @use(%[[MOVED]])
    %slot = "udonhl.ptr_add"(%names, %c4) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    %second = "udonhl.load"(%slot) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    func.call @use(%second) : (!udonhl.ptr<0>) -> ()
    // An unrelocated word is the integer its bytes hold (here null).
    // CHECK: %[[ZERO:.*]] = arith.constant 0 : i32
    // CHECK-NEXT: %[[NULL:.*]] = "udonhl.inttoptr"(%[[ZERO]])
    // CHECK-NEXT: call @use(%[[NULL]])
    %end = "udonhl.ptr_add"(%names, %c8) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    %terminator = "udonhl.load"(%end) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    func.call @use(%terminator) : (!udonhl.ptr<0>) -> ()
    // A dynamic offset, a table that may be written and a Unity buffer
    // token are left alone.
    // CHECK: %[[DYNAMIC:.*]] = "udonhl.ptr_add"(%{{.*}}, %arg0)
    // CHECK-NEXT: "udonhl.load"(%[[DYNAMIC]])
    %any = "udonhl.ptr_add"(%names, %dynamic) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    %anything = "udonhl.load"(%any) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    func.call @use(%anything) : (!udonhl.ptr<0>) -> ()
    // CHECK: %[[MUTABLE:.*]] = udonhl.address_of @mutable_names
    // CHECK-NEXT: "udonhl.load"(%[[MUTABLE]])
    %mutable = udonhl.address_of @mutable_names : <0>
    %written = "udonhl.load"(%mutable) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    func.call @use(%written) : (!udonhl.ptr<0>) -> ()
    // CHECK: %[[TOKENS:.*]] = udonhl.address_of @tokens
    // CHECK-NEXT: "udonhl.load"(%[[TOKENS]])
    %tokens = udonhl.address_of @tokens : <0>
    %token = "udonhl.load"(%tokens) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    func.call @use(%token) : (!udonhl.ptr<0>) -> ()
    return
  }
}
