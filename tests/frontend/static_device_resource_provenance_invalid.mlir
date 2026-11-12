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
