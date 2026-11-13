module {
  func.func private @clCreateBuffer() -> !udonhl.ptr<0>
  func.func private @udonclDeviceBufferAdopt(!udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>) -> i32
  func.func private @udonclDeviceBufferView(!udonhl.ptr<0>, i32, i32, i32, !udonhl.ptr<0>) -> i32
  func.func private @udonclBufferViewResolve(!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
  func.func private @udonclResolvePhysicalBuffer(!udonhl.ptr<0>, !udonhl.ptr<0>) -> i32

  func.func private @resolve_inner(%view: !udonhl.ptr<0>, %raw: !udonhl.ptr<0>, %offset: !udonhl.ptr<0>, %length: !udonhl.ptr<0>) -> i32 {
    %status = func.call @udonclBufferViewResolve(%view, %raw, %offset, %length) : (!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    return %status : i32
  }

  func.func private @resolve_outer(%view: !udonhl.ptr<0>, %raw: !udonhl.ptr<0>, %offset: !udonhl.ptr<0>, %length: !udonhl.ptr<0>) -> i32 {
    %zero = arith.constant 0 : i32
    %three = arith.constant 3 : i32
    %status = func.call @resolve_inner(%view, %raw, %offset, %length) : (!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    %invalid = arith.cmpi eq, %status, %three : i32
    cf.cond_br %invalid, ^invalid, ^normalize
  ^invalid:
    cf.br ^done(%three : i32)
  ^normalize:
    %failed = arith.cmpi ne, %status, %zero : i32
    %normalized = arith.extui %failed : i1 to i32
    cf.br ^done(%normalized : i32)
  ^done(%result: i32):
    return %result : i32
  }

  func.func @ResolveThroughWrappers() attributes {udoncl.export} {
    %zero = arith.constant 0 : i32
    %four = arith.constant 4 : i32
    %eight = arith.constant 8 : i32
    %twenty = arith.constant 20 : i32
    %elements = arith.constant 16 : i32
    %bytes = arith.constant 64 : i32
    %dtype = arith.constant 6 : i32
    %context = "udonhl.inttoptr"(%zero) : (i32) -> !udonhl.ptr<0>
    %resource = func.call @clCreateBuffer() : () -> !udonhl.ptr<0>
    %resource_address = "udonhl.ptrtoint"(%resource) : (!udonhl.ptr<0>) -> i32
    %null_address = "udonhl.ptrtoint"(%context) : (!udonhl.ptr<0>) -> i32
    %resource_is_null = arith.cmpi eq, %resource_address, %null_address : i32
    %handle = "udonhl.alloca"(%eight) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %adopted = func.call @udonclDeviceBufferAdopt(%context, %resource, %bytes, %handle) : (!udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>) -> i32
    %view = "udonhl.alloca"(%twenty) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %viewed = func.call @udonclDeviceBufferView(%handle, %zero, %elements, %dtype, %view) : (!udonhl.ptr<0>, i32, i32, i32, !udonhl.ptr<0>) -> i32
    %raw = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %offset = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %length = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %token = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %status = func.call @resolve_outer(%view, %raw, %offset, %length) : (!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    %ok = arith.cmpi eq, %status, %zero : i32
    cf.cond_br %ok, ^success, ^failure
  ^success:
    %raw_address = "udonhl.load"(%raw) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> i32
    %raw_buffer = "udonhl.inttoptr"(%raw_address) : (i32) -> !udonhl.ptr<0>
    %resolved = func.call @udonclResolvePhysicalBuffer(%raw_buffer, %token) : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    return
  ^failure:
    return
  }
}
