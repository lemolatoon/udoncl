// A resource stored in a cl_mem array that is also indexed at a run-time
// index is an error naming the array.
module {
  "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, initial_value = array<i8: 0, 0, 0, 0, 0, 0, 0, 0>, linkage = "internal", storage_type = !llvm.array<2 x ptr>, sym_name = "buffers"}> : () -> ()
  func.func private @clCreateBuffer() -> !udonhl.ptr<0>
  func.func private @clReleaseMemObject(!udonhl.ptr<0>) -> i32
  func.func @Initialize() attributes {udoncl.export} {
    %table = udonhl.address_of @buffers : !udonhl.ptr<0>
    %resource = func.call @clCreateBuffer() : () -> !udonhl.ptr<0>
    // expected-error@+1 {{static device resource stored in the cl_mem array 'buffers', which @Dispose indexes at a run-time index or accesses partially}}
    "udonhl.store"(%resource, %table) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    return
  }
  func.func @Dispose(%index: i32) attributes {udoncl.export} {
    %c4 = arith.constant 4 : i32
    %table = udonhl.address_of @buffers : !udonhl.ptr<0>
    %offset = arith.muli %index, %c4 : i32
    // expected-note@+1 {{accessed here}}
    %cell = "udonhl.ptr_add"(%table, %offset) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    %resource = "udonhl.load"(%cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    %status = func.call @clReleaseMemObject(%resource) : (!udonhl.ptr<0>) -> i32
    return
  }
}

// -----

// A partial access of an element (two bytes of it) is an error as well.
module {
  "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, initial_value = array<i8: 0, 0, 0, 0, 0, 0, 0, 0>, linkage = "internal", storage_type = !llvm.array<2 x ptr>, sym_name = "buffers"}> : () -> ()
  func.func private @clCreateBuffer() -> !udonhl.ptr<0>
  func.func @Initialize() -> i16 attributes {udoncl.export} {
    %c4 = arith.constant 4 : i32
    %c6 = arith.constant 6 : i32
    %table = udonhl.address_of @buffers : !udonhl.ptr<0>
    %cell = "udonhl.ptr_add"(%table, %c4) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    %resource = func.call @clCreateBuffer() : () -> !udonhl.ptr<0>
    // expected-error@+1 {{static device resource stored in the cl_mem array 'buffers', which @Initialize indexes at a run-time index or accesses partially}}
    "udonhl.store"(%resource, %cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    %half = "udonhl.ptr_add"(%table, %c6) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    // expected-note@+1 {{accessed here}}
    %bits = "udonhl.load"(%half) <{alignment = 2 : i64}> : (!udonhl.ptr<0>) -> i16
    return %bits : i16
  }
}
