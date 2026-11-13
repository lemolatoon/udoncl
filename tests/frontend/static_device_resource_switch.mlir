module {
  func.func private @clCreateBuffer() -> i32
  func.func private @clReleaseMemObject(i32) -> i32

  func.func @RunSwitch(%selector: i32) attributes {udoncl.export} {
    %first = func.call @clCreateBuffer() : () -> i32
    %second = func.call @clCreateBuffer() : () -> i32
    cf.switch %selector : i32, [
      default: ^merge(%first : i32),
      0: ^merge(%second : i32),
      1: ^forward(%first : i32)
    ]
  ^forward(%value: i32):
    cf.br ^merge(%value : i32)
  ^merge(%selected: i32):
    %status = func.call @clReleaseMemObject(%selected) : (i32) -> i32
    return
  }
}
