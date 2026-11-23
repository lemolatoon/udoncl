// RUN: %vrc-opt --split-input-file --vrc-unroll-static-resource-loops %s | %FileCheck %s

// A rotated constant-trip loop that creates a buffer per iteration becomes
// one allocation per iteration with a constant induction variable.
// CHECK-LABEL: func.func @Initialize
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : i32
// CHECK: call @clCreateBuffer(%[[C0]])
// CHECK: call @clCreateBuffer(%[[C1]])
// CHECK: call @clCreateBuffer(%[[C2]])
// CHECK-NOT: cf.cond_br
// CHECK: return
module {
  func.func private @clCreateBuffer(i32) -> !udonhl.ptr<0>
  func.func private @keep(!udonhl.ptr<0>)
  func.func @Initialize() attributes {udoncl.export} {
    %c0 = arith.constant 0 : i32
    cf.br ^loop(%c0 : i32)
  ^loop(%i: i32):
    %buffer = func.call @clCreateBuffer(%i) : (i32) -> !udonhl.ptr<0>
    func.call @keep(%buffer) : (!udonhl.ptr<0>) -> ()
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

// A loop body of several blocks with an early exit, reached through a
// helper that creates the buffer, unrolls the same way; the exit edge keeps
// its operands.
// CHECK-LABEL: func.func @Initialize
// CHECK-COUNT-2: call @create
// CHECK-NOT: call @create
// CHECK-NOT: ^bb{{.*}}(%{{.*}}: i32, %{{.*}}: i32)
// CHECK: return
module {
  func.func private @clCreateBuffer(i32) -> !udonhl.ptr<0>
  func.func private @status() -> i32
  func.func @create(%bytes: i32) -> !udonhl.ptr<0> {
    %buffer = func.call @clCreateBuffer(%bytes) : (i32) -> !udonhl.ptr<0>
    return %buffer : !udonhl.ptr<0>
  }
  func.func @Initialize() -> i32 attributes {udoncl.export} {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    cf.br ^loop(%c0, %c0 : i32, i32)
  ^loop(%i: i32, %last: i32):
    %buffer = func.call @create(%c64) : (i32) -> !udonhl.ptr<0>
    %status = func.call @status() : () -> i32
    %failed = arith.cmpi ne, %status, %c0 : i32
    cf.cond_br %failed, ^done(%status : i32), ^latch
  ^latch:
    %c1 = arith.constant 1 : i32
    %next = arith.addi %i, %c1 : i32
    %c2 = arith.constant 2 : i32
    %stop = arith.cmpi eq, %next, %c2 : i32
    cf.cond_br %stop, ^done(%status : i32), ^loop(%next, %status : i32, i32)
  ^done(%result: i32):
    return %result : i32
  }
}

// -----

// Loops without a Device allocation, with a bound that is not a constant,
// or with more iterations than the limit stay for discovery to judge.
// CHECK-LABEL: func.func @NoAllocation
// CHECK: cf.cond_br
// CHECK-LABEL: func.func @DynamicBound
// CHECK: cf.cond_br
// CHECK-LABEL: func.func @TooMany
// CHECK: cf.cond_br
module {
  func.func private @clCreateBuffer(i32) -> !udonhl.ptr<0>
  func.func private @work(i32)
  func.func @NoAllocation() attributes {udoncl.export} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c3 = arith.constant 3 : i32
    cf.br ^loop(%c0 : i32)
  ^loop(%i: i32):
    func.call @work(%i) : (i32) -> ()
    %next = arith.addi %i, %c1 : i32
    %more = arith.cmpi slt, %next, %c3 : i32
    cf.cond_br %more, ^loop(%next : i32), ^done
  ^done:
    return
  }
  func.func @DynamicBound(%count: i32) attributes {udoncl.export} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    cf.br ^loop(%c0 : i32)
  ^loop(%i: i32):
    %buffer = func.call @clCreateBuffer(%i) : (i32) -> !udonhl.ptr<0>
    %next = arith.addi %i, %c1 : i32
    %more = arith.cmpi slt, %next, %count : i32
    cf.cond_br %more, ^loop(%next : i32), ^done
  ^done:
    return
  }
  func.func @TooMany() attributes {udoncl.export} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c100 = arith.constant 100 : i32
    cf.br ^loop(%c0 : i32)
  ^loop(%i: i32):
    %buffer = func.call @clCreateBuffer(%i) : (i32) -> !udonhl.ptr<0>
    %next = arith.addi %i, %c1 : i32
    %more = arith.cmpi slt, %next, %c100 : i32
    cf.cond_br %more, ^loop(%next : i32), ^done
  ^done:
    return
  }
}

// -----

// clang merges an early return of the body into the latch condition, enters
// the loop from a conditional branch and uses the last status after it: the
// data exit stays on each copy, the induction exit decides the last copy,
// and the exit block receives the status of the copy that left.
// CHECK-LABEL: func.func @Initialize
// CHECK-COUNT-3: call @clCreateBuffer
// CHECK-NOT: call @clCreateBuffer
// CHECK: ^bb{{[0-9]+}}(%[[STATUS:.*]]: i32):
// CHECK: return %[[STATUS]] : i32
module {
  func.func private @clCreateBuffer(i32) -> !udonhl.ptr<0>
  func.func private @status() -> i32
  func.func @Initialize(%ready: i1) -> i32 attributes {udoncl.export} {
    %c0 = arith.constant 0 : i32
    %c9 = arith.constant 9 : i32
    cf.cond_br %ready, ^loop(%c0 : i32), ^failed(%c9 : i32)
  ^loop(%i: i32):
    %buffer = func.call @clCreateBuffer(%i) : (i32) -> !udonhl.ptr<0>
    %status = func.call @status() : () -> i32
    %failed = arith.cmpi ne, %status, %c0 : i32
    %c1 = arith.constant 1 : i32
    %next = arith.addi %i, %c1 : i32
    %last = arith.cmpi ugt, %i, %c1 : i32
    %true = arith.constant true
    %leave = arith.select %failed, %true, %last : i1
    cf.cond_br %leave, ^done, ^loop(%next : i32)
  ^done:
    return %status : i32
  ^failed(%code: i32):
    return %code : i32
  }
}
