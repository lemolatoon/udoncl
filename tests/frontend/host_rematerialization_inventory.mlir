module attributes {
  vrc.unity_buffer_pool = [
    {id = 7 : i32},
    {id = 3 : i32}
  ]
} {
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
