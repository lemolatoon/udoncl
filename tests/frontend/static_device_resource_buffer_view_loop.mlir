module {
  func.func private @clCreateBuffer() -> !udonhl.ptr<0>
  func.func private @udonclDeviceBufferAdopt(!udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>) -> i32
  func.func private @udonclDeviceBufferView(!udonhl.ptr<0>, i32, i32, i32, !udonhl.ptr<0>) -> i32
  func.func private @udononnxCopyBufferView(!udonhl.ptr<0>, !udonhl.ptr<0>)
  func.func private @udonclBufferViewResolve(!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32

  func.func @ResolveFiniteViewArray() attributes {udoncl.export, vrc.export_id = 1 : i32} {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %two = arith.constant 2 : i32
    %four = arith.constant 4 : i32
    %eight = arith.constant 8 : i32
    %twenty = arith.constant 20 : i32
    %eighty_eight = arith.constant 88 : i32
    %one_hundred_seventy_six = arith.constant 176 : i32
    %elements = arith.constant 16 : i32
    %dtype = arith.constant 6 : i32
    %bytes = arith.constant 64 : i32
    %context = "udonhl.inttoptr"(%zero) : (i32) -> !udonhl.ptr<0>
    %resource = func.call @clCreateBuffer() : () -> !udonhl.ptr<0>
    %handle = "udonhl.alloca"(%eight) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %adopted = func.call @udonclDeviceBufferAdopt(%context, %resource, %bytes, %handle) : (!udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>) -> i32
    %view = "udonhl.alloca"(%twenty) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %viewed = func.call @udonclDeviceBufferView(%handle, %zero, %elements, %dtype, %view) : (!udonhl.ptr<0>, i32, i32, i32, !udonhl.ptr<0>) -> i32
    %views = "udonhl.alloca"(%one_hundred_seventy_six) <{alignment = 8 : i64}> : (i32) -> !udonhl.ptr<0>
    func.call @udononnxCopyBufferView(%views, %view) : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    %second = "udonhl.ptr_add"(%views, %eighty_eight) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    func.call @udononnxCopyBufferView(%second, %view) : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    %raw = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %offset = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %length = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    cf.br ^loop(%zero : i32)
  ^loop(%index: i32):
    %byte_offset = arith.muli %index, %eighty_eight : i32
    %current = "udonhl.ptr_add"(%views, %byte_offset) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    %resolved = func.call @udonclBufferViewResolve(%current, %raw, %offset, %length) : (!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    %next = arith.addi %index, %one overflow<nuw> : i32
    %continue = arith.cmpi ult, %next, %two : i32
    cf.cond_br %continue, ^loop(%next : i32), ^done
  ^done:
    return
  }
}
