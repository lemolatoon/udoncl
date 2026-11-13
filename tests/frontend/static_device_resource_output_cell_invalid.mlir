module {
  func.func private @clCreateBuffer() -> !udonhl.ptr<0>

  func.func @RejectOrderedResourceComparison() attributes {udoncl.export} {
    %zero = arith.constant 0 : i32
    %resource = func.call @clCreateBuffer() : () -> !udonhl.ptr<0>
    %address = "udonhl.ptrtoint"(%resource) : (!udonhl.ptr<0>) -> i32
    %ordered = arith.cmpi ult, %address, %zero : i32
    return
  }
}
