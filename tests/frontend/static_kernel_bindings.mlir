module {
  func.func private @clCreateBuffer() -> i32
  func.func private @enqueue(%input: i32, %output: i32, %scale: i32) -> i32 attributes {
    vrc.device_dispatch_contract = {
      kernel_id = 7 : i32,
      profile = "fixture",
      resource_operands = [
        {access = "read", index = 0 : i32},
        {access = "write", index = 1 : i32}
      ],
      scalar_operands = array<i32: 2>
    },
    vrc.device_resource_operands = [
      {access = "read", capture = "none", index = 0 : i32},
      {access = "write", capture = "none", index = 1 : i32}
    ]
  }

  func.func @Run() attributes {udoncl.export} {
    %input = func.call @clCreateBuffer() : () -> i32
    %first_output = func.call @clCreateBuffer() : () -> i32
    %second_output = func.call @clCreateBuffer() : () -> i32
    %first_scale = arith.constant 3 : i32
    %second_scale = arith.constant 5 : i32
    %first = func.call @enqueue(%input, %first_output, %first_scale) : (i32, i32, i32) -> i32
    %second = func.call @enqueue(%input, %second_output, %second_scale) : (i32, i32, i32) -> i32
    return
  }
}
