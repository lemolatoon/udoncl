module {
  func.func private @clCreateBuffer(i32, i64, i64, i32) -> i32
  func.func @Run(%bytes: i64) attributes {vrc.export_id = 1 : i32} {
    %c0 = arith.constant 0 : i32
    %flags = arith.constant 1 : i64
    // expected-error @below {{static allocation flags and capacity must be constant and valid}}
    %buffer = func.call @clCreateBuffer(%c0, %flags, %bytes, %c0) : (i32, i64, i64, i32) -> i32
    return
  }
}

// -----

module {
  func.func private @clCreateBuffer(i32, i64, i64, i32) -> i32
  func.func @Run() attributes {vrc.export_id = 1 : i32} {
    %c0 = arith.constant 0 : i32
    %flags = arith.constant 1 : i64
    %bytes = arith.constant 0 : i64
    // expected-error @below {{static allocation flags and capacity must be constant and valid}}
    %buffer = func.call @clCreateBuffer(%c0, %flags, %bytes, %c0) : (i32, i64, i64, i32) -> i32
    return
  }
}
