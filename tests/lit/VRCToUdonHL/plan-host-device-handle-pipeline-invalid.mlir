// RUN: %vrc-opt --split-input-file --verify-diagnostics --vrc-plan-host-device-handle-pipeline %s

// An allocation loop of more iterations than the unroll limit has no
// finite set of allocation contexts: discovery rejects it, so no plan
// undercounts it.
module {
  func.func private @clCreateBuffer(!udonhl.ptr<0>, i64, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> !udonhl.ptr<0>
  func.func private @keep(!udonhl.ptr<0>)
  func.func @Initialize(%context: !udonhl.ptr<0>) attributes {udoncl.export} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c65 = arith.constant 65 : i32
    %flags = arith.constant 1 : i64
    %c16 = arith.constant 16 : i32
    %null = "udonhl.inttoptr"(%c0) : (i32) -> !udonhl.ptr<0>
    cf.br ^loop(%c0 : i32)
  ^loop(%index: i32):
    // expected-error @below {{static device allocation in an unproved loop is unsupported}}
    %buffer = func.call @clCreateBuffer(%context, %flags, %c16, %null, %null) : (!udonhl.ptr<0>, i64, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> !udonhl.ptr<0>
    func.call @keep(%buffer) : (!udonhl.ptr<0>) -> ()
    %next = arith.addi %index, %c1 : i32
    %more = arith.cmpi slt, %next, %c65 : i32
    cf.cond_br %more, ^loop(%next : i32), ^done
  ^done:
    return
  }
}
