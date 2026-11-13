module {
  "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, initial_value = array<i8: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>, linkage = "internal", storage_type = !llvm.array<16 x i8>, sym_name = "global_handle"}> : () -> ()
  "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, initial_value = array<i8: 0, 0, 0, 0, 0, 0, 0, 0>, linkage = "internal", storage_type = !llvm.struct<"struct.udonclBufferHandle", (i32, i32)>, sym_name = "struct_handle"}> : () -> ()

  func.func private @clCreateBuffer() -> !udonhl.ptr<0>
  func.func private @clReleaseMemObject(!udonhl.ptr<0>) -> i32
  func.func private @udonclDeviceBufferAdopt(!udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>) -> i32
  func.func private @udonclDeviceBufferGetRaw(!udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
  func.func private @udonclDeviceBufferDestroy(!udonhl.ptr<0>) -> i32

  func.func private @adopt(%resource: !udonhl.ptr<0>, %handle: !udonhl.ptr<0>) -> i32 {
    %zero = arith.constant 0 : i32
    %bytes = arith.constant 64 : i32
    %context = "udonhl.inttoptr"(%zero) : (i32) -> !udonhl.ptr<0>
    %status = func.call @udonclDeviceBufferAdopt(%context, %resource, %bytes, %handle) : (!udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>) -> i32
    %ok = arith.cmpi eq, %status, %zero : i32
    cf.cond_br %ok, ^success, ^failure
  ^success:
    return %zero : i32
  ^failure:
    return %status : i32
  }

  func.func @AGlobalInitialize() attributes {udoncl.export} {
    %eight = arith.constant 8 : i32
    %resource = func.call @clCreateBuffer() : () -> !udonhl.ptr<0>
    %base = udonhl.address_of @global_handle : !udonhl.ptr<0>
    %handle = "udonhl.ptr_add"(%base, %eight) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    %status = func.call @adopt(%resource, %handle) : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    return
  }

  func.func @BGlobalUse() attributes {udoncl.export} {
    %zero = arith.constant 0 : i32
    %four = arith.constant 4 : i32
    %eight = arith.constant 8 : i32
    %base = udonhl.address_of @global_handle : !udonhl.ptr<0>
    %handle = "udonhl.ptr_add"(%base, %eight) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    %raw_cell = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %null = "udonhl.inttoptr"(%zero) : (i32) -> !udonhl.ptr<0>
    "udonhl.store"(%null, %raw_cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    %status = func.call @udonclDeviceBufferGetRaw(%handle, %raw_cell) : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    %raw = "udonhl.load"(%raw_cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    %released = func.call @clReleaseMemObject(%raw) : (!udonhl.ptr<0>) -> i32
    %destroyed = func.call @udonclDeviceBufferDestroy(%handle) : (!udonhl.ptr<0>) -> i32
    return
  }

  func.func @CLocalUse() attributes {udoncl.export} {
    %zero = arith.constant 0 : i32
    %eight = arith.constant 8 : i32
    %four = arith.constant 4 : i32
    %handle = "udonhl.alloca"(%eight) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %resource = func.call @clCreateBuffer() : () -> !udonhl.ptr<0>
    %adopted = func.call @adopt(%resource, %handle) : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    %raw_cell = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %null = "udonhl.inttoptr"(%zero) : (i32) -> !udonhl.ptr<0>
    "udonhl.store"(%null, %raw_cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    %status = func.call @udonclDeviceBufferGetRaw(%handle, %raw_cell) : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    %raw = "udonhl.load"(%raw_cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    %released = func.call @clReleaseMemObject(%raw) : (!udonhl.ptr<0>) -> i32
    %destroyed = func.call @udonclDeviceBufferDestroy(%handle) : (!udonhl.ptr<0>) -> i32
    return
  }

  func.func @DStructInitialize() attributes {udoncl.export} {
    %resource = func.call @clCreateBuffer() : () -> !udonhl.ptr<0>
    %handle = udonhl.address_of @struct_handle : !udonhl.ptr<0>
    %status = func.call @adopt(%resource, %handle) : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    return
  }

  func.func @EStructUse() attributes {udoncl.export} {
    %zero = arith.constant 0 : i32
    %four = arith.constant 4 : i32
    %handle = udonhl.address_of @struct_handle : !udonhl.ptr<0>
    %raw_cell = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %null = "udonhl.inttoptr"(%zero) : (i32) -> !udonhl.ptr<0>
    "udonhl.store"(%null, %raw_cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    %status = func.call @udonclDeviceBufferGetRaw(%handle, %raw_cell) : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    %raw = "udonhl.load"(%raw_cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    %released = func.call @clReleaseMemObject(%raw) : (!udonhl.ptr<0>) -> i32
    %destroyed = func.call @udonclDeviceBufferDestroy(%handle) : (!udonhl.ptr<0>) -> i32
    return
  }
}
