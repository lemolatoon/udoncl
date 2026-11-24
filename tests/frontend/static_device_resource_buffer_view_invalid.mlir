module {
  func.func private @clCreateBuffer() -> !udonhl.ptr<0>
  func.func private @udonclDeviceBufferAdopt(!udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>) -> i32
  func.func private @udonclBufferViewResolve(!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32

  func.func @MismatchedHandleFields() attributes {udoncl.export} {
    %zero = arith.constant 0 : i32
    %four = arith.constant 4 : i32
    %eight = arith.constant 8 : i32
    %twelve = arith.constant 12 : i32
    %sixteen = arith.constant 16 : i32
    %twenty = arith.constant 20 : i32
    %bytes = arith.constant 64 : i32
    %dtype = arith.constant 6 : i32
    %context = "udonhl.inttoptr"(%zero) : (i32) -> !udonhl.ptr<0>
    %resource_a = func.call @clCreateBuffer() : () -> !udonhl.ptr<0>
    %resource_b = func.call @clCreateBuffer() : () -> !udonhl.ptr<0>
    %handle_a = "udonhl.alloca"(%eight) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %handle_b = "udonhl.alloca"(%eight) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %adopted_a = func.call @udonclDeviceBufferAdopt(%context, %resource_a, %bytes, %handle_a) : (!udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>) -> i32
    %adopted_b = func.call @udonclDeviceBufferAdopt(%context, %resource_b, %bytes, %handle_b) : (!udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>) -> i32
    %slot_a = "udonhl.load"(%handle_a) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> i32
    %handle_b4 = "udonhl.ptr_add"(%handle_b, %four) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    %generation_b = "udonhl.load"(%handle_b4) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> i32
    %view = "udonhl.alloca"(%twenty) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    "udonhl.store"(%slot_a, %view) <{alignment = 4 : i64}> : (i32, !udonhl.ptr<0>) -> ()
    %view4 = "udonhl.ptr_add"(%view, %four) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    "udonhl.store"(%generation_b, %view4) <{alignment = 4 : i64}> : (i32, !udonhl.ptr<0>) -> ()
    %view8 = "udonhl.ptr_add"(%view, %eight) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    "udonhl.store"(%zero, %view8) <{alignment = 4 : i64}> : (i32, !udonhl.ptr<0>) -> ()
    %view12 = "udonhl.ptr_add"(%view, %twelve) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    "udonhl.store"(%bytes, %view12) <{alignment = 4 : i64}> : (i32, !udonhl.ptr<0>) -> ()
    %view16 = "udonhl.ptr_add"(%view, %sixteen) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    "udonhl.store"(%dtype, %view16) <{alignment = 4 : i64}> : (i32, !udonhl.ptr<0>) -> ()
    %raw = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %offset = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %length = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %resolved = func.call @udonclBufferViewResolve(%view, %raw, %offset, %length) : (!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    return
  }
}
