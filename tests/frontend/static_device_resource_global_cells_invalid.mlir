module {
  "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, linkage = "external", storage_type = !udonhl.ptr<0>, sym_name = "external_cell"}> : () -> ()
  func.func private @clCreateBuffer() -> !udonhl.ptr<0>
  func.func @Run() attributes {udoncl.export} {
    %resource = func.call @clCreateBuffer() : () -> !udonhl.ptr<0>
    %address = udonhl.address_of @external_cell : !udonhl.ptr<0>
    // expected-error@+1 {{static device resource stored through an unsupported or escaping global cell}}
    "udonhl.store"(%resource, %address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    return
  }
}

// -----

module {
  "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, initial_value = 0 : i32, linkage = "internal", storage_type = !udonhl.ptr<0>, sym_name = "escaped_cell"}> : () -> ()
  func.func private @clCreateBuffer() -> !udonhl.ptr<0>
  func.func private @escape(!udonhl.ptr<0>)
  func.func @Run() attributes {udoncl.export} {
    %resource = func.call @clCreateBuffer() : () -> !udonhl.ptr<0>
    %address = udonhl.address_of @escaped_cell : !udonhl.ptr<0>
    func.call @escape(%address) : (!udonhl.ptr<0>) -> ()
    // expected-error@+1 {{static device resource stored through an unsupported or escaping global cell}}
    "udonhl.store"(%resource, %address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    return
  }
}

// -----

module {
  "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, initial_value = 0 : i32, linkage = "internal", storage_type = !udonhl.ptr<0>, sym_name = "unknown_cell"}> : () -> ()
  func.func private @clCreateBuffer() -> !udonhl.ptr<0>
  func.func private @clReleaseMemObject(!udonhl.ptr<0>) -> i32
  func.func @AInitialize() attributes {udoncl.export} {
    %resource = func.call @clCreateBuffer() : () -> !udonhl.ptr<0>
    %address = udonhl.address_of @unknown_cell : !udonhl.ptr<0>
    "udonhl.store"(%resource, %address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    return
  }
  func.func @BOverwrite() attributes {udoncl.export} {
    %seven = arith.constant 7 : i32
    %unknown = "udonhl.inttoptr"(%seven) : (i32) -> !udonhl.ptr<0>
    %address = udonhl.address_of @unknown_cell : !udonhl.ptr<0>
    "udonhl.store"(%unknown, %address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    return
  }
  func.func @CRun() attributes {udoncl.export} {
    %address = udonhl.address_of @unknown_cell : !udonhl.ptr<0>
    %resource = "udonhl.load"(%address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    // expected-error@+1 {{resource use has unresolved non-handle provenance}}
    %status = func.call @clReleaseMemObject(%resource) : (!udonhl.ptr<0>) -> i32
    return
  }
}
