module {
  func.func private @clCreateBuffer() -> i32

  func.func private @allocate() -> i32 {
    %buffer = func.call @clCreateBuffer() : () -> i32
    return %buffer : i32
  }

  func.func private @initialize() {
    %first = func.call @allocate() : () -> i32
    %second = func.call @allocate() : () -> i32
    return
  }

  func.func @Run() attributes {udoncl.export} {
    func.call @initialize() : () -> ()
    return
  }
}
