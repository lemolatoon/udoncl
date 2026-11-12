module {
  func.func private @clCreateBuffer() -> i32
  func.func private @clReleaseMemObject(i32) -> i32

  func.func @identity(%buffer: i32) -> i32 {
    return %buffer : i32
  }

  func.func @RunLoop(%condition: i1) attributes {udoncl.export} {
    %first = func.call @clCreateBuffer() : () -> i32
    %second = func.call @clCreateBuffer() : () -> i32
    cf.br ^loop(%first : i32)
  ^loop(%carried: i32):
    %alias = func.call @identity(%carried) : (i32) -> i32
    cf.cond_br %condition, ^backedge, ^exit
  ^backedge:
    cf.br ^loop(%second : i32)
  ^exit:
    %status = func.call @clReleaseMemObject(%alias) : (i32) -> i32
    return
  }
}
