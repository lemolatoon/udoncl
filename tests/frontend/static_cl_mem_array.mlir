// A `static cl_mem buffers[3]` array: created in a loop, released in a
// loop and read at constant elements. Once vrc-unroll-static-resource-loops
// has unrolled the loops, every access is a whole element at a constant
// offset and each element is a tracked cell of its own.
module {
  "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, initial_value = array<i8: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>, linkage = "internal", storage_type = !llvm.array<3 x ptr>, sym_name = "buffers"}> : () -> ()
  func.func private @clCreateBuffer() -> !udonhl.ptr<0>
  func.func private @clReleaseMemObject(!udonhl.ptr<0>) -> i32
  func.func private @read(!udonhl.ptr<0>) -> i32 attributes {vrc.device_resource_operands = [{access = "read", capture = "none", index = 0 : i32}]}

  // Stores through an out-parameter, as `create(&buffers[i])` does.
  func.func @create(%cell: !udonhl.ptr<0>) {
    %resource = func.call @clCreateBuffer() : () -> !udonhl.ptr<0>
    "udonhl.store"(%resource, %cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    return
  }

  func.func @AInitialize() attributes {udoncl.export} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c4 = arith.constant 4 : i32
    %table = udonhl.address_of @buffers : !udonhl.ptr<0>
    cf.br ^loop(%c0 : i32)
  ^loop(%i: i32):
    %offset = arith.muli %i, %c4 : i32
    %cell = "udonhl.ptr_add"(%table, %offset) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    %resource = func.call @clCreateBuffer() : () -> !udonhl.ptr<0>
    "udonhl.store"(%resource, %cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    %next = arith.addi %i, %c1 : i32
    %more = arith.cmpi slt, %next, %c2 : i32
    cf.cond_br %more, ^loop(%next : i32), ^last
  ^last:
    %c8 = arith.constant 8 : i32
    %third = "udonhl.ptr_add"(%table, %c8) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    func.call @create(%third) : (!udonhl.ptr<0>) -> ()
    return
  }

  // Reads the second element.
  func.func @BRun() attributes {udoncl.export} {
    %c4 = arith.constant 4 : i32
    %table = udonhl.address_of @buffers : !udonhl.ptr<0>
    %cell = "udonhl.ptr_add"(%table, %c4) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    %resource = "udonhl.load"(%cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    %status = func.call @read(%resource) : (!udonhl.ptr<0>) -> i32
    return
  }

  func.func @ZDispose() attributes {udoncl.export} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c3 = arith.constant 3 : i32
    %c4 = arith.constant 4 : i32
    %null = "udonhl.inttoptr"(%c0) : (i32) -> !udonhl.ptr<0>
    %table = udonhl.address_of @buffers : !udonhl.ptr<0>
    cf.br ^loop(%c0 : i32)
  ^loop(%i: i32):
    %offset = arith.muli %i, %c4 : i32
    %cell = "udonhl.ptr_add"(%table, %offset) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    %resource = "udonhl.load"(%cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    %status = func.call @clReleaseMemObject(%resource) : (!udonhl.ptr<0>) -> i32
    "udonhl.store"(%null, %cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    %next = arith.addi %i, %c1 : i32
    %more = arith.cmpi slt, %next, %c3 : i32
    cf.cond_br %more, ^loop(%next : i32), ^done
  ^done:
    return
  }
}
