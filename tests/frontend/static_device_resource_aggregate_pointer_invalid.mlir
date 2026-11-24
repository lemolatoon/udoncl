module {
  "udonhl.global"() <{address_space = 0 : i32, alignment = 8 : i64, linkage = "internal", storage_type = !llvm.array<40 x i8>, sym_name = "session"}> : () -> ()
  "udonhl.global"() <{address_space = 0 : i32, alignment = 8 : i64, linkage = "internal", storage_type = !llvm.array<88 x i8>, sym_name = "bindings"}> : () -> ()

  func.func private @clCreateBuffer() -> !udonhl.ptr<0>
  func.func private @udonclDeviceBufferAdopt(!udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>) -> i32
  func.func private @udonclDeviceBufferView(!udonhl.ptr<0>, i32, i32, i32, !udonhl.ptr<0>) -> i32

  func.func @Run(%condition: i1) attributes {udoncl.export} {
    %zero = arith.constant 0 : i32
    %eight = arith.constant 8 : i32
    %elements = arith.constant 16 : i32
    %dtype = arith.constant 6 : i32
    %bytes = arith.constant 64 : i32
    %context = "udonhl.inttoptr"(%zero) : (i32) -> !udonhl.ptr<0>
    %resource = func.call @clCreateBuffer() : () -> !udonhl.ptr<0>
    %handle = "udonhl.alloca"(%eight) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %adopted = func.call @udonclDeviceBufferAdopt(%context, %resource, %bytes, %handle) : (!udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>) -> i32
    %session = udonhl.address_of @session : !udonhl.ptr<0>
    %bindings = udonhl.address_of @bindings : !udonhl.ptr<0>
    %bindings_cell = "udonhl.ptr_add"(%session, %eight) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    cf.cond_br %condition, ^initialize, ^skip
  ^initialize:
    "udonhl.store"(%bindings, %bindings_cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    cf.br ^use
  ^skip:
    cf.br ^use
  ^use:
    %loaded = "udonhl.load"(%bindings_cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    %viewed = func.call @udonclDeviceBufferView(%handle, %zero, %elements, %dtype, %loaded) : (!udonhl.ptr<0>, i32, i32, i32, !udonhl.ptr<0>) -> i32
    return
  }
}
