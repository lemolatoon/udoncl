module {
  func.func private @clCreateBuffer() -> i32
  // expected-error @below {{invalid scalar operand in Device dispatch contract}}
  func.func private @enqueue(i32) -> i32 attributes {
    vrc.device_dispatch_contract = {
      kernel_id = 7 : i32,
      profile = "fixture",
      resource_operands = [{access = "write", index = 0 : i32}],
      scalar_operands = array<i32: 0>
    },
    vrc.device_resource_operands = [{access = "write", capture = "none", index = 0 : i32}]
  }
  func.func @Run() attributes {udoncl.export} {
    %buffer = func.call @clCreateBuffer() : () -> i32
    %status = func.call @enqueue(%buffer) : (i32) -> i32
    return
  }
}
