module {
  func.func private @clCreateBuffer() -> !udonhl.ptr<0>
  func.func private @clReleaseMemObject(!udonhl.ptr<0>) -> i32
  func.func private @udonclDeviceBufferAdopt(!udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>) -> i32
  func.func private @udonclDeviceBufferView(!udonhl.ptr<0>, i32, i32, i32, !udonhl.ptr<0>) -> i32
  func.func private @udonclBufferViewResolve(!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32

  func.func private @udononnxCopyBufferView(%destination: !udonhl.ptr<0>, %source: !udonhl.ptr<0>) {
    %four = arith.constant 4 : i32
    %eight = arith.constant 8 : i32
    %twelve = arith.constant 12 : i32
    %sixteen = arith.constant 16 : i32
    %source4 = "udonhl.ptr_add"(%source, %four) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    %destination4 = "udonhl.ptr_add"(%destination, %four) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    %source8 = "udonhl.ptr_add"(%source, %eight) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    %destination8 = "udonhl.ptr_add"(%destination, %eight) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    %source12 = "udonhl.ptr_add"(%source, %twelve) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    %destination12 = "udonhl.ptr_add"(%destination, %twelve) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    %source16 = "udonhl.ptr_add"(%source, %sixteen) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    %destination16 = "udonhl.ptr_add"(%destination, %sixteen) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    return
  }

  func.func @Run() attributes {udoncl.export} {
    %zero = arith.constant 0 : i32
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
    %tensor = "udonhl.alloca"(%eighty_eight) <{alignment = 8 : i64}> : (i32) -> !udonhl.ptr<0>
    func.call @udononnxCopyBufferView(%tensor, %view) : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    %raw_cell = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %null = "udonhl.inttoptr"(%zero) : (i32) -> !udonhl.ptr<0>
    "udonhl.store"(%null, %raw_cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    %offset_cell = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %length_cell = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %resolved = func.call @udonclBufferViewResolve(%tensor, %raw_cell, %offset_cell, %length_cell) : (!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    %raw = "udonhl.load"(%raw_cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    %released = func.call @clReleaseMemObject(%raw) : (!udonhl.ptr<0>) -> i32
    return
  }
}
