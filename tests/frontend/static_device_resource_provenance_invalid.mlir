module {
  func.func private @clCreateBuffer() -> i32
  func.func private @unknown(i32)
  func.func @Run() attributes {udoncl.export} {
    %buffer = func.call @clCreateBuffer() : () -> i32
    // expected-error @below {{static device resource escapes to an unknown external call}}
    func.call @unknown(%buffer) : (i32) -> ()
    return
  }
}

// -----

module {
  func.func private @udonclCreateBufferForQueue() -> !udonhl.ptr<0>
  func.func private @udonclResolvePhysicalBuffer(!udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
  func.func @Run() attributes {udoncl.export} {
    %buffer = func.call @udonclCreateBufferForQueue() : () -> !udonhl.ptr<0>
    %output_resource = func.call @udonclCreateBufferForQueue() : () -> !udonhl.ptr<0>
    // expected-error @below {{static device resource passed as udonclResolvePhysicalBuffer output cell}}
    %status = func.call @udonclResolvePhysicalBuffer(%buffer, %output_resource) : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    return
  }
}

// -----

module {
  func.func private @udonclCreateBufferForQueue() -> !udonhl.ptr<0>
  func.func private @udonclResolvePhysicalBuffer(!udonhl.ptr<0>, i32) -> i32
  func.func @Run() attributes {udoncl.export} {
    %buffer = func.call @udonclCreateBufferForQueue() : () -> !udonhl.ptr<0>
    %token = arith.constant 0 : i32
    // expected-error @below {{udonclResolvePhysicalBuffer requires signature (!udonhl.ptr<0>, !udonhl.ptr<0>) -> i32}}
    %status = func.call @udonclResolvePhysicalBuffer(%buffer, %token) : (!udonhl.ptr<0>, i32) -> i32
    return
  }
}

// -----

module {
  func.func private @udonclCreateBufferForQueue() -> !udonhl.ptr<0>
  func.func private @udonclResolvePhysicalBuffer(%buffer: !udonhl.ptr<0>, %token: !udonhl.ptr<0>) -> i32 {
    %zero = arith.constant 0 : i32
    return %zero : i32
  }
  func.func @Run() attributes {udoncl.export} {
    %buffer = func.call @udonclCreateBufferForQueue() : () -> !udonhl.ptr<0>
    %four = arith.constant 4 : i32
    %token = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    // expected-error @below {{udonclResolvePhysicalBuffer requires an unlinked external declaration during static resource analysis}}
    %status = func.call @udonclResolvePhysicalBuffer(%buffer, %token) : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    return
  }
}

// -----

module {
  func.func private @udonclCreateBufferForQueue() -> !udonhl.ptr<0>
  func.func private @udonclResolvePhysicalBuffer(!udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
  func.func @Run(%condition: i1) attributes {udoncl.export} {
    %buffer = func.call @udonclCreateBufferForQueue() : () -> !udonhl.ptr<0>
    %one = arith.constant 1 : i32
    %scalar_pointer = "udonhl.inttoptr"(%one) : (i32) -> !udonhl.ptr<0>
    %unknown = arith.select %condition, %buffer, %scalar_pointer : !udonhl.ptr<0>
    %four = arith.constant 4 : i32
    %token = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    // expected-error @below {{resource use has unresolved non-handle provenance}}
    %status = func.call @udonclResolvePhysicalBuffer(%unknown, %token) : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    return
  }
}

// -----

module {
  func.func private @clCreateBuffer() -> i32
  func.func @Run() attributes {udoncl.export} {
    %buffer = func.call @clCreateBuffer() : () -> i32
    %slot = memref.alloca() : memref<i32>
    // expected-error @below {{unsupported operation consumes a static device resource}}
    memref.store %buffer, %slot[] : memref<i32>
    return
  }
}

// -----

module {
  func.func private @clCreateBuffer() -> i32
  func.func @Run(%condition: i1) attributes {udoncl.export} {
    // expected-error @below {{unsupported region or terminator prevents complete device resource provenance}}
    scf.if %condition {
      %buffer = func.call @clCreateBuffer() : () -> i32
    }
    return
  }
}

// -----

module {
  func.func private @clCreateBuffer()
  func.func @Run() attributes {udoncl.export} {
    // expected-error @below {{static device allocation must return exactly one handle}}
    func.call @clCreateBuffer() : () -> ()
    return
  }
}

// -----

module {
  func.func private @clCreateBuffer() -> i32
  func.func @Run() -> i32 attributes {udoncl.export} {
    %buffer = func.call @clCreateBuffer() : () -> i32
    // expected-error @below {{static device resource escapes through an exported result}}
    return %buffer : i32
  }
}

// -----

module {
  func.func private @clCreateBuffer() -> i32
  func.func @Run() attributes {udoncl.export} {
    %buffer = func.call @clCreateBuffer() : () -> i32
    %one = arith.constant 1 : i32
    // expected-error @below {{unsupported operation consumes a static device resource}}
    %bad = arith.addi %buffer, %one : i32
    return
  }
}

// -----

module {
  func.func private @clCreateBuffer() -> i32
  func.func private @clReleaseMemObject(i32) -> i32
  func.func @Run(%condition: i1) attributes {udoncl.export} {
    %buffer = func.call @clCreateBuffer() : () -> i32
    %one = arith.constant 1 : i32
    cf.cond_br %condition, ^left, ^right
  ^left:
    cf.br ^merge(%buffer : i32)
  ^right:
    cf.br ^merge(%one : i32)
  ^merge(%value: i32):
    // expected-error @below {{resource use has unresolved non-handle provenance}}
    %status = func.call @clReleaseMemObject(%value) : (i32) -> i32
    return
  }
}
