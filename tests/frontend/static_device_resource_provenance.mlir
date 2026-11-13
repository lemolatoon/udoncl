module {
  func.func private @clCreateBuffer() -> i32
  func.func private @clReleaseMemObject(i32) -> i32
  func.func private @__vrc_enqueue_probe_device_args(i32) -> i32 attributes {
    vrc.device_resource_operands = [{access = "read", capture = "none", index = 0 : i32}]
  }

  func.func @make() -> i32 {
    %buffer = func.call @clCreateBuffer() : () -> i32
    return %buffer : i32
  }

  func.func @identity(%buffer: i32) -> i32 {
    return %buffer : i32
  }

  func.func @Run(%condition: i1) attributes {udoncl.export} {
    %first = func.call @make() : () -> i32
    %second = func.call @make() : () -> i32
    %first_alias = func.call @identity(%first) : (i32) -> i32
    %second_alias = func.call @identity(%second) : (i32) -> i32
    %selected = arith.select %condition, %first_alias, %second_alias : i32
    %null = arith.constant 0 : i32
    %nullable = arith.select %condition, %first_alias, %null : i32
    %nullable_is_null = arith.cmpi eq, %nullable, %null : i32
    cf.cond_br %condition, ^left, ^right
  ^left:
    cf.br ^merge(%first_alias : i32)
  ^right:
    cf.br ^merge(%second_alias : i32)
  ^merge(%joined: i32):
    %status0 = func.call @clReleaseMemObject(%selected) : (i32) -> i32
    %status1 = func.call @clReleaseMemObject(%joined) : (i32) -> i32
    %status2 = func.call @clReleaseMemObject(%nullable) : (i32) -> i32
    %status3 = func.call @__vrc_enqueue_probe_device_args(%first_alias) : (i32) -> i32
    return
  }

  func.func @RunUnknown(%condition: i1) attributes {udoncl.export} {
    %buffer = func.call @clCreateBuffer() : () -> i32
    %one = arith.constant 1 : i32
    cf.cond_br %condition, ^resource, ^scalar
  ^resource:
    cf.br ^merge(%buffer : i32)
  ^scalar:
    cf.br ^merge(%one : i32)
  ^merge(%value: i32):
    %dead = func.call @identity(%value) : (i32) -> i32
    return
  }
}
