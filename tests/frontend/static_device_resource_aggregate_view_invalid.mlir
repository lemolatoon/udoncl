module {
  func.func private @clCreateBuffer() -> !udonhl.ptr<0>
  func.func private @udonclDeviceBufferAdopt(!udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>) -> i32
  func.func private @udonclDeviceBufferView(!udonhl.ptr<0>, i32, i32, i32, !udonhl.ptr<0>) -> i32
  func.func private @udonclBufferViewResolve(!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32

  func.func @Run() attributes {udoncl.export} {
    %zero = arith.constant 0 : i32
    %four = arith.constant 4 : i32
    %eight = arith.constant 8 : i32
    %twenty = arith.constant 20 : i32
    %elements = arith.constant 16 : i32
    %dtype = arith.constant 6 : i32
    %bytes = arith.constant 64 : i32
    %context = "udonhl.inttoptr"(%zero) : (i32) -> !udonhl.ptr<0>
    %view = "udonhl.alloca"(%twenty) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %raw_cell = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %offset_cell = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %length_cell = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %resolved = func.call @udonclBufferViewResolve(%view, %raw_cell, %offset_cell, %length_cell) : (!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    %resource = func.call @clCreateBuffer() : () -> !udonhl.ptr<0>
    %handle = "udonhl.alloca"(%eight) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %adopted = func.call @udonclDeviceBufferAdopt(%context, %resource, %bytes, %handle) : (!udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>) -> i32
    %viewed = func.call @udonclDeviceBufferView(%handle, %zero, %elements, %dtype, %view) : (!udonhl.ptr<0>, i32, i32, i32, !udonhl.ptr<0>) -> i32
    return
  }
}
