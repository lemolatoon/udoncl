// RUN: %vrc-opt --split-input-file --vrc-plan-host-device-handle-pipeline %s | %FileCheck %s

// Every allocation of a Host is one handle with the capacity of its
// clCreateBuffer, in allocation context order.

// CHECK-LABEL: module attributes
// CHECK-SAME: vrc.host_device_handle_plan = {buffer_bytes = array<i64: 128, 64>, handles = 2 : i64}
module {
  func.func private @clCreateBuffer(!udonhl.ptr<0>, i64, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> !udonhl.ptr<0>
  func.func private @keep(!udonhl.ptr<0>)
  func.func @Initialize(%context: !udonhl.ptr<0>) attributes {udoncl.export} {
    %flags = arith.constant 1 : i64
    %c128 = arith.constant 128 : i32
    %c64 = arith.constant 64 : i32
    %zero = arith.constant 0 : i32
    %null = "udonhl.inttoptr"(%zero) : (i32) -> !udonhl.ptr<0>
    %first = func.call @clCreateBuffer(%context, %flags, %c128, %null, %null) : (!udonhl.ptr<0>, i64, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> !udonhl.ptr<0>
    func.call @keep(%first) : (!udonhl.ptr<0>) -> ()
    %second = func.call @clCreateBuffer(%context, %flags, %c64, %null, %null) : (!udonhl.ptr<0>, i64, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> !udonhl.ptr<0>
    func.call @keep(%second) : (!udonhl.ptr<0>) -> ()
    return
  }
}

// -----

// A helper that allocates two buffers, called in a 22-trip loop, is 44
// handles; a clCreateBuffer directly in a 4-trip loop of another export is
// 4 more.

// CHECK-LABEL: module attributes
// CHECK-SAME: vrc.host_device_handle_plan = {buffer_bytes = array<i64: 256, 512, 256, 512, 256, 512, 256, 512, 256, 512, 256, 512, 256, 512, 256, 512, 256, 512, 256, 512, 256, 512, 256, 512, 256, 512, 256, 512, 256, 512, 256, 512, 256, 512, 256, 512, 256, 512, 256, 512, 256, 512, 256, 512, 16, 16, 16, 16>, handles = 48 : i64}
module {
  func.func private @clCreateBuffer(!udonhl.ptr<0>, i64, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> !udonhl.ptr<0>
  func.func private @keep(!udonhl.ptr<0>)
  func.func @create_layer(%context: !udonhl.ptr<0>) {
    %flags = arith.constant 1 : i64
    %c256 = arith.constant 256 : i32
    %c512 = arith.constant 512 : i32
    %zero = arith.constant 0 : i32
    %null = "udonhl.inttoptr"(%zero) : (i32) -> !udonhl.ptr<0>
    %key = func.call @clCreateBuffer(%context, %flags, %c256, %null, %null) : (!udonhl.ptr<0>, i64, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> !udonhl.ptr<0>
    func.call @keep(%key) : (!udonhl.ptr<0>) -> ()
    %value = func.call @clCreateBuffer(%context, %flags, %c512, %null, %null) : (!udonhl.ptr<0>, i64, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> !udonhl.ptr<0>
    func.call @keep(%value) : (!udonhl.ptr<0>) -> ()
    return
  }
  func.func @Initialize(%context: !udonhl.ptr<0>) attributes {udoncl.export} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c22 = arith.constant 22 : i32
    cf.br ^layers(%c0 : i32)
  ^layers(%layer: i32):
    func.call @create_layer(%context) : (!udonhl.ptr<0>) -> ()
    %next = arith.addi %layer, %c1 : i32
    %more = arith.cmpi slt, %next, %c22 : i32
    cf.cond_br %more, ^layers(%next : i32), ^done
  ^done:
    return
  }
  func.func @Scratch(%context: !udonhl.ptr<0>) attributes {udoncl.export} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %flags = arith.constant 1 : i64
    %c16 = arith.constant 16 : i32
    %null = "udonhl.inttoptr"(%c0) : (i32) -> !udonhl.ptr<0>
    cf.br ^scratch(%c0 : i32)
  ^scratch(%index: i32):
    %buffer = func.call @clCreateBuffer(%context, %flags, %c16, %null, %null) : (!udonhl.ptr<0>, i64, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> !udonhl.ptr<0>
    func.call @keep(%buffer) : (!udonhl.ptr<0>) -> ()
    %next = arith.addi %index, %c1 : i32
    %more = arith.cmpi slt, %next, %c4 : i32
    cf.cond_br %more, ^scratch(%next : i32), ^done
  ^done:
    return
  }
}

// -----

// One helper called from two exports allocates in two contexts. The
// capacity of each comes from its own call path.

// CHECK-LABEL: module attributes
// CHECK-SAME: vrc.host_device_handle_plan = {buffer_bytes = array<i64: 32, 48>, handles = 2 : i64}
module {
  func.func private @clCreateBuffer(!udonhl.ptr<0>, i64, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> !udonhl.ptr<0>
  func.func private @keep(!udonhl.ptr<0>)
  func.func @create(%context: !udonhl.ptr<0>, %bytes: i32) {
    %flags = arith.constant 1 : i64
    %zero = arith.constant 0 : i32
    %null = "udonhl.inttoptr"(%zero) : (i32) -> !udonhl.ptr<0>
    %buffer = func.call @clCreateBuffer(%context, %flags, %bytes, %null, %null) : (!udonhl.ptr<0>, i64, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> !udonhl.ptr<0>
    func.call @keep(%buffer) : (!udonhl.ptr<0>) -> ()
    return
  }
  func.func @First(%context: !udonhl.ptr<0>) attributes {udoncl.export} {
    %c32 = arith.constant 32 : i32
    func.call @create(%context, %c32) : (!udonhl.ptr<0>, i32) -> ()
    return
  }
  func.func @Second(%context: !udonhl.ptr<0>) attributes {udoncl.export} {
    %c48 = arith.constant 48 : i32
    func.call @create(%context, %c48) : (!udonhl.ptr<0>, i32) -> ()
    return
  }
}

// -----

// A static constructor configured as a root allocates like an export; a
// function that is no configured root does not count.

// CHECK-LABEL: module attributes
// CHECK-SAME: vrc.host_device_handle_plan = {buffer_bytes = array<i64: 24, 8>, handles = 2 : i64}
module {
  func.func private @clCreateBuffer(!udonhl.ptr<0>, i64, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> !udonhl.ptr<0>
  func.func private @keep(!udonhl.ptr<0>)
  func.func @allocate(%bytes: i32) {
    %flags = arith.constant 1 : i64
    %zero = arith.constant 0 : i32
    %null = "udonhl.inttoptr"(%zero) : (i32) -> !udonhl.ptr<0>
    %buffer = func.call @clCreateBuffer(%null, %flags, %bytes, %null, %null) : (!udonhl.ptr<0>, i64, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> !udonhl.ptr<0>
    func.call @keep(%buffer) : (!udonhl.ptr<0>) -> ()
    return
  }
  func.func @Setup() attributes {vrc.export_id = 1 : i32, vrc.static_constructor} {
    %c8 = arith.constant 8 : i32
    func.call @allocate(%c8) : (i32) -> ()
    return
  }
  func.func @Run() attributes {vrc.export_id = 2 : i32} {
    %c24 = arith.constant 24 : i32
    func.call @allocate(%c24) : (i32) -> ()
    return
  }
  func.func @Unused() {
    %c40 = arith.constant 40 : i32
    func.call @allocate(%c40) : (i32) -> ()
    return
  }
}

// -----

// A Host that allocates nothing needs no handle.

// CHECK-LABEL: module attributes
// CHECK-SAME: vrc.host_device_handle_plan = {buffer_bytes = array<i64>, handles = 0 : i64}
module {
  func.func @Run() -> i32 attributes {udoncl.export} {
    %c0 = arith.constant 0 : i32
    return %c0 : i32
  }
}
