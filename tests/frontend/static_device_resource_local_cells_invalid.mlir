module {
  func.func private @clCreateBuffer() -> !udonhl.ptr<0>
  func.func @Run() attributes {udoncl.export} {
    %eight = arith.constant 8 : i32
    %buffer = func.call @clCreateBuffer() : () -> !udonhl.ptr<0>
    %cell = "udonhl.alloca"(%eight) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    // expected-error @below {{static device resource stored through a non-singleton or escaping local cell}}
    "udonhl.store"(%buffer, %cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    return
  }
}

// -----

module {
  func.func private @clCreateBuffer() -> !udonhl.ptr<0>
  func.func @Run() attributes {udoncl.export} {
    %four = arith.constant 4 : i32
    %buffer = func.call @clCreateBuffer() : () -> !udonhl.ptr<0>
    %cell = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    "udonhl.store"(%cell, %cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    // expected-error @below {{static device resource stored through a non-singleton or escaping local cell}}
    "udonhl.store"(%buffer, %cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    return
  }
}

// -----

module {
  func.func private @clCreateBuffer() -> !udonhl.ptr<0>
  func.func @Run() attributes {udoncl.export} {
    %buffer = func.call @clCreateBuffer() : () -> !udonhl.ptr<0>
    // expected-error @below {{static device resource used as a Host memory address}}
    %loaded = "udonhl.load"(%buffer) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    return
  }
}

// -----

module {
  func.func private @clCreateBuffer() -> !udonhl.ptr<0>
  func.func @Run() attributes {udoncl.export} {
    %zero = arith.constant 0 : i32
    %null = "udonhl.inttoptr"(%zero) : (i32) -> !udonhl.ptr<0>
    %buffer = func.call @clCreateBuffer() : () -> !udonhl.ptr<0>
    // expected-error @below {{static device resource used as a Host memory address}}
    "udonhl.store"(%null, %buffer) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    return
  }
}

// -----

module {
  func.func private @clCreateBuffer() -> !udonhl.ptr<0>
  func.func @Run() attributes {udoncl.export} {
    %four = arith.constant 4 : i32
    %buffer = func.call @clCreateBuffer() : () -> !udonhl.ptr<0>
    %cell = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    // expected-error @below {{static device resource stored through a non-singleton or escaping local cell}}
    "udonhl.store"(%buffer, %cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    %mixed = "udonhl.load"(%cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> i32
    return
  }
}

// -----

module {
  func.func private @clCreateBuffer() -> !udonhl.ptr<0>
  func.func private @clReleaseMemObject(!udonhl.ptr<0>) -> i32
  func.func @Run() attributes {udoncl.export} {
    %four = arith.constant 4 : i32
    %one = arith.constant 1 : i32
    %unknown = "udonhl.inttoptr"(%one) : (i32) -> !udonhl.ptr<0>
    %buffer = func.call @clCreateBuffer() : () -> !udonhl.ptr<0>
    %cell = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    "udonhl.store"(%buffer, %cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    "udonhl.store"(%unknown, %cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    %loaded = "udonhl.load"(%cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    // expected-error @below {{resource use has unresolved non-handle provenance}}
    %status = func.call @clReleaseMemObject(%loaded) : (!udonhl.ptr<0>) -> i32
    return
  }
}

// -----

module {
  func.func private @clCreateBuffer() -> !udonhl.ptr<0>
  func.func private @escape(!udonhl.ptr<0>)
  func.func @Run() attributes {udoncl.export} {
    %four = arith.constant 4 : i32
    %buffer = func.call @clCreateBuffer() : () -> !udonhl.ptr<0>
    %cell = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    // expected-error @below {{static device resource stored through a non-singleton or escaping local cell}}
    "udonhl.store"(%buffer, %cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    func.call @escape(%cell) : (!udonhl.ptr<0>) -> ()
    return
  }
}

// -----

module {
  func.func private @clCreateBuffer() -> !udonhl.ptr<0>
  func.func private @clReleaseMemObject(!udonhl.ptr<0>) -> i32
  func.func @Run(%condition: i1) attributes {udoncl.export} {
    %four = arith.constant 4 : i32
    %buffer = func.call @clCreateBuffer() : () -> !udonhl.ptr<0>
    %cell = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    cf.cond_br %condition, ^initialized, ^missing
  ^initialized:
    "udonhl.store"(%buffer, %cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    cf.br ^merge
  ^missing:
    cf.br ^merge
  ^merge:
    %loaded = "udonhl.load"(%cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    // expected-error @below {{resource use has unresolved non-handle provenance}}
    %status = func.call @clReleaseMemObject(%loaded) : (!udonhl.ptr<0>) -> i32
    return
  }
}
