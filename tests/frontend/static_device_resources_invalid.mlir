module {
  func.func private @clCreateBuffer() -> i32

  // expected-error @+1 {{reachable resource-producing recursion is unsupported}}
  func.func @self_recursive() {
    %buffer = func.call @clCreateBuffer() : () -> i32
    func.call @self_recursive() : () -> ()
    return
  }

  func.func @RunSelfRecursive() attributes {vrc.export_id = 1 : i32} {
    func.call @self_recursive() : () -> ()
    return
  }
}

// -----

module {
  func.func private @clCreateBuffer() -> i32

  // expected-error @+1 {{reachable resource-producing recursion is unsupported}}
  func.func @mutual_a() {
    func.call @mutual_b() : () -> ()
    return
  }

  func.func @mutual_b() {
    %buffer = func.call @clCreateBuffer() : () -> i32
    func.call @mutual_a() : () -> ()
    return
  }

  func.func @RunMutualRecursive() attributes {vrc.export_id = 1 : i32} {
    func.call @mutual_a() : () -> ()
    return
  }
}

// -----

module {
  func.func private @unknown()

  func.func @RunIndirect() attributes {vrc.export_id = 1 : i32} {
    %callee = func.constant @unknown : () -> ()
    // expected-error @+1 {{reachable indirect call prevents complete static device resource discovery}}
    func.call_indirect %callee() : () -> ()
    return
  }
}

// -----

module {
  func.func private @unknown()

  func.func @helper_with_indirect() {
    %callee = func.constant @unknown : () -> ()
    // expected-error @+1 {{reachable indirect call prevents complete static device resource discovery}}
    func.call_indirect %callee() : () -> ()
    return
  }

  func.func @RunIndirectHelper() attributes {vrc.export_id = 1 : i32} {
    func.call @helper_with_indirect() : () -> ()
    return
  }
}

// -----

module {
  func.func private @clCreateBuffer() -> i32

  func.func @RunStructuredLoop() attributes {vrc.export_id = 1 : i32} {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    scf.for %index = %zero to %one step %one {
      // expected-error @+1 {{static device allocation in an unproved loop is unsupported}}
      %buffer = func.call @clCreateBuffer() : () -> i32
    }
    return
  }
}

// -----

module {
  func.func private @clCreateBuffer() -> i32

  func.func @allocate() {
    %buffer = func.call @clCreateBuffer() : () -> i32
    return
  }

  func.func @RunStructuredHelperLoop() attributes {vrc.export_id = 1 : i32} {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    scf.for %index = %zero to %one step %one {
      // expected-error @+1 {{resource-producing call in an unproved loop is unsupported}}
      func.call @allocate() : () -> ()
    }
    return
  }
}

// -----

module {
  func.func private @clCreateBuffer() -> i32

  func.func @allocate() {
    %buffer = func.call @clCreateBuffer() : () -> i32
    return
  }

  func.func @RunCfgLoop(%condition: i1) attributes {vrc.export_id = 1 : i32} {
    cf.br ^loop
  ^loop:
    // expected-error @+1 {{resource-producing call in an unproved loop is unsupported}}
    func.call @allocate() : () -> ()
    cf.cond_br %condition, ^loop, ^exit
  ^exit:
    return
  }
}

// -----

module {
  func.func private @clCreateBuffer() -> i32

  func.func @RunIrreducible(%first: i1, %again: i1) attributes {vrc.export_id = 1 : i32} {
    cf.cond_br %first, ^left, ^right
  ^left:
    // expected-error @+1 {{static device allocation in an unproved loop is unsupported}}
    %buffer = func.call @clCreateBuffer() : () -> i32
    cf.br ^right
  ^right:
    cf.cond_br %again, ^left, ^exit
  ^exit:
    return
  }
}

// -----

module {
  func.func @clCreateBuffer() -> i32 {
    %zero = arith.constant 0 : i32
    return %zero : i32
  }

  func.func @RunLinkedRuntime() attributes {vrc.export_id = 1 : i32} {
    // expected-error @+1 {{static device resource discovery requires an unlinked OpenCL allocation declaration}}
    %buffer = func.call @clCreateBuffer() : () -> i32
    return
  }
}
