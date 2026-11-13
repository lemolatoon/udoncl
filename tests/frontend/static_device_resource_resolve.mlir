module {
  func.func private @udonclCreateBufferForQueue() -> !udonhl.ptr<0>
  func.func private @udonclResolvePhysicalBuffer(!udonhl.ptr<0>, !udonhl.ptr<0>) -> i32

  func.func @Run() attributes {udoncl.export} {
    %buffer = func.call @udonclCreateBufferForQueue() : () -> !udonhl.ptr<0>
    %four = arith.constant 4 : i32
    %token = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %status = func.call @udonclResolvePhysicalBuffer(%buffer, %token) : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    return
  }
}
