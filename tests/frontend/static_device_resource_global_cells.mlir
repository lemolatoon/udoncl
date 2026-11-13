module {
  "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, initial_value = 0 : i32, linkage = "internal", storage_type = !udonhl.ptr<0>, sym_name = "primary"}> : () -> ()
  "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, initial_value = 0 : i32, linkage = "internal", storage_type = !udonhl.ptr<0>, sym_name = "forwarded"}> : () -> ()
  "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, initial_value = 0 : i32, linkage = "internal", storage_type = !udonhl.ptr<0>, sym_name = "ordinary_pointer"}> : () -> ()
  func.func private @clCreateBuffer() -> !udonhl.ptr<0>
  func.func private @clReleaseMemObject(!udonhl.ptr<0>) -> i32
  func.func private @createOrdinaryObject() -> !udonhl.ptr<0>
  func.func private @consumeOrdinaryObject(!udonhl.ptr<0>)

  func.func @AInitialize() attributes {udoncl.export} {
    %resource = func.call @clCreateBuffer() : () -> !udonhl.ptr<0>
    %address = udonhl.address_of @primary : !udonhl.ptr<0>
    "udonhl.store"(%resource, %address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    return
  }

  func.func @BInitializeAgain() attributes {udoncl.export} {
    %resource = func.call @clCreateBuffer() : () -> !udonhl.ptr<0>
    %address = udonhl.address_of @primary : !udonhl.ptr<0>
    "udonhl.store"(%resource, %address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    return
  }

  func.func @CForward() attributes {udoncl.export} {
    %source = udonhl.address_of @primary : !udonhl.ptr<0>
    %resource = "udonhl.load"(%source) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    %destination = udonhl.address_of @forwarded : !udonhl.ptr<0>
    "udonhl.store"(%resource, %destination) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    return
  }

  func.func @DRun() attributes {udoncl.export} {
    %address = udonhl.address_of @forwarded : !udonhl.ptr<0>
    %resource = "udonhl.load"(%address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    %status = func.call @clReleaseMemObject(%resource) : (!udonhl.ptr<0>) -> i32
    return
  }

  func.func @EOrdinaryPointer() attributes {udoncl.export} {
    %object = func.call @createOrdinaryObject() : () -> !udonhl.ptr<0>
    %address = udonhl.address_of @ordinary_pointer : !udonhl.ptr<0>
    "udonhl.store"(%object, %address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    %loaded = "udonhl.load"(%address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    func.call @consumeOrdinaryObject(%loaded) : (!udonhl.ptr<0>) -> ()
    return
  }

  func.func @ZDispose() attributes {udoncl.export} {
    %zero = arith.constant 0 : i32
    %null = "udonhl.inttoptr"(%zero) : (i32) -> !udonhl.ptr<0>
    %primary_address = udonhl.address_of @primary : !udonhl.ptr<0>
    %forwarded_address = udonhl.address_of @forwarded : !udonhl.ptr<0>
    "udonhl.store"(%null, %primary_address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    "udonhl.store"(%null, %forwarded_address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    return
  }
}
