// RUN: %vrc-opt --split-input-file --vrc-unroll-static-resource-loops %s | %FileCheck %s

// A bind loop whose iterations return early after every bind, as
// `if (status != SUCCESS) return status;` compiles: one return block takes
// the status of every exit. The latch comes before the header in the
// region, and the last bind uses a value of an earlier block of the body
// that comes after it: each copy is cloned in dominance order.
// CHECK-LABEL: func.func @Bind
// CHECK-COUNT-10: call @udononnxBindInput
// CHECK-NOT: call @udononnxBindInput
// CHECK: ^bb{{[0-9]+}}(%[[STATUS:.*]]: i32):
// CHECK-NEXT: return %[[STATUS]] : i32
module {
  func.func private @udononnxBindInput(!udonhl.ptr<0>, i32) -> i32
  func.func private @buffer(i32) -> !udonhl.ptr<0>
  func.func @Bind(%session: !udonhl.ptr<0>) -> i32 attributes {udoncl.export} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %first = func.call @udononnxBindInput(%session, %c0) : (!udonhl.ptr<0>, i32) -> i32
    %ok = arith.cmpi eq, %first, %c0 : i32
    cf.cond_br %ok, ^header(%c0 : i32), ^exit(%first : i32)
  ^latch:
    %next = arith.addi %i, %c1 : i32
    %more = arith.cmpi ult, %i, %c2 : i32
    cf.cond_br %more, ^header(%next : i32), ^exit(%c0 : i32)
  ^last:
    %s3 = func.call @udononnxBindInput(%session, %index) : (!udonhl.ptr<0>, i32) -> i32
    %f3 = arith.cmpi eq, %s3, %c0 : i32
    cf.cond_br %f3, ^latch, ^exit(%s3 : i32)
  ^header(%i: i32):
    %s1 = func.call @udononnxBindInput(%session, %i) : (!udonhl.ptr<0>, i32) -> i32
    %f1 = arith.cmpi eq, %s1, %c0 : i32
    cf.cond_br %f1, ^second, ^exit(%s1 : i32)
  ^second:
    %index = arith.addi %i, %c2 : i32
    %s2 = func.call @udononnxBindInput(%session, %index) : (!udonhl.ptr<0>, i32) -> i32
    %f2 = arith.cmpi eq, %s2, %c0 : i32
    cf.cond_br %f2, ^last, ^exit(%s2 : i32)
  ^exit(%status: i32):
    return %status : i32
  }
}

// -----

// Early returns to several distinct return blocks unroll the same way.
// CHECK-LABEL: func.func @Bind
// CHECK-COUNT-4: call @udononnxBindInput
// CHECK-NOT: call @udononnxBindInput
module {
  func.func private @udononnxBindInput(!udonhl.ptr<0>, i32) -> i32
  func.func @Bind(%session: !udonhl.ptr<0>) -> i32 attributes {udoncl.export} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c7 = arith.constant 7 : i32
    cf.br ^header(%c0 : i32)
  ^header(%i: i32):
    %s1 = func.call @udononnxBindInput(%session, %i) : (!udonhl.ptr<0>, i32) -> i32
    %f1 = arith.cmpi ne, %s1, %c0 : i32
    cf.cond_br %f1, ^failed_input, ^output
  ^output:
    %s2 = func.call @udononnxBindInput(%session, %i) : (!udonhl.ptr<0>, i32) -> i32
    %f2 = arith.cmpi ne, %s2, %c0 : i32
    cf.cond_br %f2, ^failed_output, ^latch
  ^latch:
    %next = arith.addi %i, %c1 : i32
    %more = arith.cmpi slt, %next, %c2 : i32
    cf.cond_br %more, ^header(%next : i32), ^done
  ^failed_input:
    return %c7 : i32
  ^failed_output:
    return %c1 : i32
  ^done:
    return %c0 : i32
  }
}

// -----

// A status of the loop used after it, with exits to several blocks, has no
// single exit block to receive it: the loop stays for static Device
// resource discovery to judge.
// CHECK-LABEL: func.func @Bind
// CHECK: ^bb{{[0-9]+}}(%{{.*}}: i32, %{{.*}}: i32):
// CHECK-COUNT-1: call @udononnxBindInput
// CHECK-NOT: call @udononnxBindInput
module {
  func.func private @udononnxBindInput(!udonhl.ptr<0>, i32) -> i32
  func.func private @report(i32)
  func.func @Bind(%session: !udonhl.ptr<0>) -> i32 attributes {udoncl.export} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    cf.br ^header(%c0, %c0 : i32, i32)
  ^header(%i: i32, %last: i32):
    %status = func.call @udononnxBindInput(%session, %i) : (!udonhl.ptr<0>, i32) -> i32
    %failed = arith.cmpi ne, %status, %c0 : i32
    cf.cond_br %failed, ^failed, ^latch
  ^latch:
    %next = arith.addi %i, %c1 : i32
    %more = arith.cmpi slt, %next, %c2 : i32
    cf.cond_br %more, ^header(%next, %status : i32, i32), ^done
  ^failed:
    func.call @report(%status) : (i32) -> ()
    return %status : i32
  ^done:
    return %status : i32
  }
}

// -----

// Two loops in a row: the copies of the first leave it only from the last
// iteration, so the second loop has one entry and unrolls too.
// CHECK-LABEL: func.func @Initialize
// CHECK-COUNT-5: call @clCreateBuffer
// CHECK-NOT: call @clCreateBuffer
// CHECK-NOT: cf.cond_br
module {
  func.func private @clCreateBuffer(i32) -> !udonhl.ptr<0>
  func.func private @keep(!udonhl.ptr<0>)
  func.func @Initialize() attributes {udoncl.export} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    cf.br ^first(%c0 : i32)
  ^first(%i: i32):
    %a = func.call @clCreateBuffer(%i) : (i32) -> !udonhl.ptr<0>
    func.call @keep(%a) : (!udonhl.ptr<0>) -> ()
    %next_i = arith.addi %i, %c1 : i32
    %more_i = arith.cmpi slt, %next_i, %c2 : i32
    cf.cond_br %more_i, ^first(%next_i : i32), ^second(%c0 : i32)
  ^second(%j: i32):
    %b = func.call @clCreateBuffer(%j) : (i32) -> !udonhl.ptr<0>
    func.call @keep(%b) : (!udonhl.ptr<0>) -> ()
    %next_j = arith.addi %j, %c1 : i32
    %more_j = arith.cmpi slt, %next_j, %c3 : i32
    cf.cond_br %more_j, ^second(%next_j : i32), ^done
  ^done:
    return
  }
}
