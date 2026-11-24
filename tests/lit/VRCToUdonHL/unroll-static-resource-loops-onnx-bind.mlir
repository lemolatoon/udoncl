// RUN: %vrc-opt --split-input-file --vrc-unroll-static-resource-loops %s | %FileCheck %s

// A constant-trip loop that binds a Device buffer to a named ONNX tensor per
// iteration names one tensor per iteration once unrolled; the table loads
// read constant offsets.
// CHECK-LABEL: func.func @Initialize
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : i32
// CHECK-DAG: %[[C8:.*]] = arith.constant 8 : i32
// CHECK: call @udononnxBindInput
// CHECK: "udonhl.ptr_add"(%{{.*}}, %[[C4]])
// CHECK: call @udononnxBindInput
// CHECK: "udonhl.ptr_add"(%{{.*}}, %[[C8]])
// CHECK: call @udononnxBindInput
// CHECK-NOT: call @udononnxBindInput
// CHECK-NOT: cf.cond_br
// CHECK: return
module {
  "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, constant, initial_value = array<i8: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>, linkage = "internal", storage_type = !llvm.array<3 x ptr>, sym_name = "names"}> : () -> ()
  func.func private @udononnxBindInput(!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
  func.func @Initialize(%session: !udonhl.ptr<0>, %memory: !udonhl.ptr<0>) attributes {udoncl.export} {
    %c0 = arith.constant 0 : i32
    %names = udonhl.address_of @names : <0>
    cf.br ^loop(%c0 : i32)
  ^loop(%i: i32):
    %c4 = arith.constant 4 : i32
    %offset = arith.muli %i, %c4 : i32
    %slot = "udonhl.ptr_add"(%names, %offset) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    %name = "udonhl.load"(%slot) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    %status = func.call @udononnxBindInput(%session, %name, %memory, %c0, %memory, %memory) : (!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    %c1 = arith.constant 1 : i32
    %next = arith.addi %i, %c1 : i32
    %c3 = arith.constant 3 : i32
    %more = arith.cmpi slt, %next, %c3 : i32
    cf.cond_br %more, ^loop(%next : i32), ^done
  ^done:
    return
  }
}

// -----

// A loop whose binds go through a helper unrolls the same way.
// CHECK-LABEL: func.func @Initialize
// CHECK-COUNT-2: call @bind
// CHECK-NOT: call @bind
// CHECK: return
module {
  func.func private @udononnxBindOutput(!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
  func.func @bind(%session: !udonhl.ptr<0>, %index: i32) {
    %status = func.call @udononnxBindOutput(%session, %session, %session, %index, %session, %session) : (!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    return
  }
  func.func @Initialize(%session: !udonhl.ptr<0>) attributes {udoncl.export} {
    %c0 = arith.constant 0 : i32
    cf.br ^loop(%c0 : i32)
  ^loop(%i: i32):
    func.call @bind(%session, %i) : (!udonhl.ptr<0>, i32) -> ()
    %c1 = arith.constant 1 : i32
    %next = arith.addi %i, %c1 : i32
    %c2 = arith.constant 2 : i32
    %more = arith.cmpi slt, %next, %c2 : i32
    cf.cond_br %more, ^loop(%next : i32), ^done
  ^done:
    return
  }
}
