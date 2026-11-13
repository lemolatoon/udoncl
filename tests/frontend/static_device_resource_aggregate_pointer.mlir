module {
  "udonhl.global"() <{address_space = 0 : i32, alignment = 8 : i64, linkage = "internal", storage_type = !llvm.array<40 x i8>, sym_name = "session"}> : () -> ()
  "udonhl.global"() <{address_space = 0 : i32, alignment = 8 : i64, linkage = "internal", storage_type = !llvm.array<176 x i8>, sym_name = "bindings"}> : () -> ()
  "udonhl.global"() <{address_space = 0 : i32, alignment = 8 : i64, linkage = "internal", storage_type = !llvm.array<40 x i8>, sym_name = "persistent_session"}> : () -> ()
  "udonhl.global"() <{address_space = 0 : i32, alignment = 8 : i64, linkage = "internal", storage_type = !llvm.array<88 x i8>, sym_name = "persistent_binding"}> : () -> ()

  func.func private @clCreateBuffer() -> !udonhl.ptr<0>
  func.func private @clReleaseMemObject(!udonhl.ptr<0>) -> i32
  func.func private @udonclDeviceBufferAdopt(!udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>) -> i32
  func.func private @udonclDeviceBufferView(!udonhl.ptr<0>, i32, i32, i32, !udonhl.ptr<0>) -> i32
  func.func private @udonclBufferViewResolve(!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32

  func.func private @udononnxCopyBufferView(%destination: !udonhl.ptr<0>, %source: !udonhl.ptr<0>) {
    return
  }

  func.func private @initializeSession(%fail: i1, %session: !udonhl.ptr<0>, %bindings: !udonhl.ptr<0>) -> i32 {
    %zero = arith.constant 0 : i32
    %error = arith.constant 7 : i32
    %eight = arith.constant 8 : i32
    %bindings_cell = "udonhl.ptr_add"(%session, %eight) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    cf.cond_br %fail, ^failure, ^success
  ^success:
    "udonhl.store"(%bindings, %bindings_cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    cf.br ^done(%zero : i32)
  ^failure:
    cf.br ^done(%error : i32)
  ^done(%status: i32):
    return %status : i32
  }

  func.func private @copyTensorView(%session: !udonhl.ptr<0>, %index: i32, %source: !udonhl.ptr<0>) {
    %eight = arith.constant 8 : i32
    %stride = arith.constant 88 : i32
    %bindings_cell = "udonhl.ptr_add"(%session, %eight) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    %bindings_base = "udonhl.load"(%bindings_cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    %offset = arith.muli %index, %stride : i32
    %tensor = "udonhl.ptr_add"(%bindings_base, %offset) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    func.call @udononnxCopyBufferView(%tensor, %source) : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    return
  }

  func.func @AInitializePersistentPointer() attributes {udoncl.export} {
    %eight = arith.constant 8 : i32
    %session = udonhl.address_of @persistent_session : !udonhl.ptr<0>
    %binding = udonhl.address_of @persistent_binding : !udonhl.ptr<0>
    %bindings_cell = "udonhl.ptr_add"(%session, %eight) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    "udonhl.store"(%binding, %bindings_cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    return
  }

  func.func @BUsePersistentPointer() attributes {udoncl.export} {
    %zero = arith.constant 0 : i32
    %four = arith.constant 4 : i32
    %eight = arith.constant 8 : i32
    %twenty = arith.constant 20 : i32
    %elements = arith.constant 16 : i32
    %dtype = arith.constant 6 : i32
    %bytes = arith.constant 64 : i32
    %context = "udonhl.inttoptr"(%zero) : (i32) -> !udonhl.ptr<0>
    %resource = func.call @clCreateBuffer() : () -> !udonhl.ptr<0>
    %handle = "udonhl.alloca"(%eight) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %adopted = func.call @udonclDeviceBufferAdopt(%context, %resource, %bytes, %handle) : (!udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>) -> i32
    %view = "udonhl.alloca"(%twenty) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %viewed = func.call @udonclDeviceBufferView(%handle, %zero, %elements, %dtype, %view) : (!udonhl.ptr<0>, i32, i32, i32, !udonhl.ptr<0>) -> i32
    %session = udonhl.address_of @persistent_session : !udonhl.ptr<0>
    %bindings_cell = "udonhl.ptr_add"(%session, %eight) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    %binding = "udonhl.load"(%bindings_cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    func.call @udononnxCopyBufferView(%binding, %view) : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    %raw_cell = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %null = "udonhl.inttoptr"(%zero) : (i32) -> !udonhl.ptr<0>
    "udonhl.store"(%null, %raw_cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    %offset_cell = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %length_cell = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %resolved = func.call @udonclBufferViewResolve(%binding, %raw_cell, %offset_cell, %length_cell) : (!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    return
  }

  func.func @Run(%condition: i1) attributes {udoncl.export} {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %four = arith.constant 4 : i32
    %eight = arith.constant 8 : i32
    %twenty = arith.constant 20 : i32
    %eighty_eight = arith.constant 88 : i32
    %elements = arith.constant 16 : i32
    %dtype = arith.constant 6 : i32
    %bytes = arith.constant 64 : i32
    %context = "udonhl.inttoptr"(%zero) : (i32) -> !udonhl.ptr<0>
    %resource = func.call @clCreateBuffer() : () -> !udonhl.ptr<0>
    %handle = "udonhl.alloca"(%eight) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %adopted = func.call @udonclDeviceBufferAdopt(%context, %resource, %bytes, %handle) : (!udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>) -> i32
    %view = "udonhl.alloca"(%twenty) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %viewed = func.call @udonclDeviceBufferView(%handle, %zero, %elements, %dtype, %view) : (!udonhl.ptr<0>, i32, i32, i32, !udonhl.ptr<0>) -> i32

    %session = udonhl.address_of @session : !udonhl.ptr<0>
    %bindings = udonhl.address_of @bindings : !udonhl.ptr<0>
    %status = func.call @initializeSession(%condition, %session, %bindings) : (i1, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    %initialized = arith.cmpi eq, %status, %zero : i32
    cf.cond_br %initialized, ^use, ^failure

  ^use:
    func.call @copyTensorView(%session, %one, %view) : (!udonhl.ptr<0>, i32, !udonhl.ptr<0>) -> ()

    %tensor = "udonhl.ptr_add"(%bindings, %eighty_eight) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    %raw_cell = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %null = "udonhl.inttoptr"(%zero) : (i32) -> !udonhl.ptr<0>
    "udonhl.store"(%null, %raw_cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    %offset_cell = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %length_cell = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %resolved = func.call @udonclBufferViewResolve(%tensor, %raw_cell, %offset_cell, %length_cell) : (!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    %raw = "udonhl.load"(%raw_cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    %released = func.call @clReleaseMemObject(%raw) : (!udonhl.ptr<0>) -> i32
    return
  ^failure:
    return
  }
}
