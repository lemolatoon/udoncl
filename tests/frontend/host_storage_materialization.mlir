module {
  func.func private @sink_i32(i32, i32)
  func.func private @sink_mixed(i8, i32)

  func.func @materialization_cases() -> i32 {
    %seed1 = arith.constant 1 : i32
    %seed2 = arith.constant 2 : i32
    %seed4 = arith.constant 4 : i32
    %seed6 = arith.constant 6 : i32
    %seed15 = arith.constant 15 : i32
    %c22 = arith.constant 22 : i32
    %c22_duplicate = arith.constant 22 : i32
    %c33 = arith.constant 33 : i32
    %narrow = arith.constant -1 : i8
    %integer = arith.constant -1 : i32
    %duplicate = arith.addi %c22, %c22_duplicate : i32
    %distinct = arith.addi %c22, %c33 : i32
    func.call @sink_i32(%c22, %c33) : (i32, i32) -> ()
    func.call @sink_mixed(%narrow, %integer) : (i8, i32) -> ()
    return %duplicate : i32
  }

  func.func @branch_literal() -> i32 {
    %c7 = arith.constant 7 : i32
    cf.br ^next(%c7 : i32)
  ^next(%value: i32):
    return %value : i32
  }
}
