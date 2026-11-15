module attributes {
  vrc.unity_buffer_pool = [
    {id = 7 : i32},
    {id = 3 : i32}
  ]
} {
  func.func @branch_literal() -> i32 {
    %c13 = arith.constant 13 : i32
    cf.br ^next(%c13 : i32)
  ^next(%value: i32):
    return %value : i32
  }

  func.func private @external_passthrough(i32) -> i32

  func.func @external_caller() -> i32 {
    %c7 = arith.constant 7 : i32
    %result = func.call @external_passthrough(%c7) : (i32) -> i32
    return %result : i32
  }

  func.func private @internal_callee(%value: i32) -> i32 {
    return %value : i32
  }

  func.func private @mixed_logical_consumer(i8, i32)

  func.func @mixed_logical_literals() {
    %narrow = arith.constant -1 : i8
    %integer = arith.constant -1 : i32
    func.call @mixed_logical_consumer(%narrow, %integer) : (i8, i32) -> ()
    return
  }

  func.func @internal_caller() -> i32 {
    %c9 = arith.constant 9 : i32
    %result = func.call @internal_callee(%c9) : (i32) -> i32
    return %result : i32
  }

  func.func @nested_direct(%value: i32, %condition: i1) -> i32 {
    %result = scf.if %condition -> (i32) {
      %c5 = arith.constant 5 : i32
      %sum = arith.addi %value, %c5 : i32
      scf.yield %sum : i32
    } else {
      scf.yield %value : i32
    }
    return %result : i32
  }

  func.func @zeta() -> (i1, i64) {
    %true = arith.constant true
    %wide = arith.constant 4294967297 : i64
    return %true, %wide : i1, i64
  }

  func.func @scratch_groups() -> i32 {
    %c11 = arith.constant 11 : i32
    %c22 = arith.constant 22 : i32
    %first = arith.addi %c11, %c22 : i32
    %duplicate = arith.addi %c11, %c11 : i32
    %c33 = arith.constant 33 : i32
    %second = arith.addi %first, %c33 : i32
    %result = arith.addi %second, %duplicate : i32
    return %result : i32
  }

  func.func @scratch_reuse() -> i32 {
    %c11 = arith.constant 11 : i32
    %c22 = arith.constant 22 : i32
    %first = arith.addi %c11, %c22 : i32
    %c33 = arith.constant 33 : i32
    %c44 = arith.constant 44 : i32
    %second = arith.addi %c33, %c44 : i32
    return %second : i32
  }

  func.func @alpha(%condition: i1) -> (i32, f32, index, i2) {
    %integer = arith.constant -1 : i32
    %floating = arith.constant -0.000000e+00 : f32
    %shared = arith.constant 42 : i32
    %narrow = arith.constant -1 : i8
    %half = arith.constant -1 : i16
    %unsupported = arith.constant 1 : i2
    %nested = scf.if %condition -> (index) {
      %index = arith.constant 42 : index
      scf.yield %index : index
    } else {
      %zero = arith.constant 0 : index
      scf.yield %zero : index
    }
    return %integer, %floating, %nested, %unsupported : i32, f32, index, i2
  }
}
