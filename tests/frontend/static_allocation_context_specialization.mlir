module {
  func.func private @clCreateBuffer() -> i32

  func.func private @allocate() -> i32 {
    %buffer = func.call @clCreateBuffer() : () -> i32
    return %buffer : i32
  }

  func.func @Run() attributes {udoncl.export} {
    %first = func.call @allocate() : () -> i32
    %second = func.call @allocate() : () -> i32
    return
  }
}
