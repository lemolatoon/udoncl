module {
  func.func private @clCreateBuffer(i32, i64, i64, i32) -> i32
  func.func private @udonclCreateBufferForQueue(i32, i64, i64, i32) -> i32

  func.func @create(%flags: i64, %elements: i32) -> i32 {
    %c0 = arith.constant 0 : i32
    %c2 = arith.constant 2 : i64
    %wide = arith.extui %elements : i32 to i64
    %bytes = arith.shli %wide, %c2 : i64
    %buffer = func.call @clCreateBuffer(%c0, %flags, %bytes, %c0) : (i32, i64, i64, i32) -> i32
    return %buffer : i32
  }

  func.func @RunA() attributes {vrc.export_id = 1 : i32} {
    %flags = arith.constant 4 : i64
    %elements = arith.constant 8 : i32
    %buffer = func.call @create(%flags, %elements) : (i64, i32) -> i32
    return
  }
  func.func @RunB() attributes {vrc.export_id = 2 : i32} {
    %flags = arith.constant 2 : i64
    %elements = arith.constant 3 : i32
    %buffer = func.call @create(%flags, %elements) : (i64, i32) -> i32
    return
  }
  func.func @RunC() attributes {vrc.export_id = 3 : i32} {
    %c0 = arith.constant 0 : i32
    %flags = arith.constant 1 : i64
    %bytes = arith.constant 64 : i64
    %buffer = func.call @udonclCreateBufferForQueue(%c0, %flags, %bytes, %c0) : (i32, i64, i64, i32) -> i32
    return
  }
}
