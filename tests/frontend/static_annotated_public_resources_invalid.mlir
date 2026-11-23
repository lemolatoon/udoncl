module {
  // expected-error@+1 {{UDONCL_PUBLIC variable image holds no Device buffer the compiler can track; store the result of one clCreateBuffer in it}}
  "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, initial_value = 0 : i32, linkage = "internal", storage_type = !udonhl.ptr<0>, sym_name = "image"}> {udoncl.public = "float32"} : () -> ()
  func.func @Run() attributes {udoncl.export} {
    %zero = arith.constant 0 : i32
    %null = "udonhl.inttoptr"(%zero) : (i32) -> !udonhl.ptr<0>
    %address = udonhl.address_of @image : !udonhl.ptr<0>
    "udonhl.store"(%null, %address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    return
  }
}

// -----

// One variable must hold the buffer of one clCreateBuffer.
module {
  // expected-error@+1 {{UDONCL_PUBLIC variable image holds the Device buffers of 2 allocation contexts; it must hold the buffer of exactly one clCreateBuffer}}
  "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, initial_value = 0 : i32, linkage = "internal", storage_type = !udonhl.ptr<0>, sym_name = "image"}> {udoncl.public = "float32"} : () -> ()
  func.func private @clCreateBuffer(!udonhl.ptr<0>, i64, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> !udonhl.ptr<0>
  func.func @AInitialize(%context: !udonhl.ptr<0>) attributes {udoncl.export} {
    %flags = arith.constant 4 : i64
    %bytes = arith.constant 64 : i32
    %zero = arith.constant 0 : i32
    %null = "udonhl.inttoptr"(%zero) : (i32) -> !udonhl.ptr<0>
    %buffer = func.call @clCreateBuffer(%context, %flags, %bytes, %null, %null) : (!udonhl.ptr<0>, i64, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> !udonhl.ptr<0>
    %address = udonhl.address_of @image : !udonhl.ptr<0>
    "udonhl.store"(%buffer, %address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    return
  }
  func.func @BReinitialize(%context: !udonhl.ptr<0>) attributes {udoncl.export} {
    %flags = arith.constant 4 : i64
    %bytes = arith.constant 64 : i32
    %zero = arith.constant 0 : i32
    %null = "udonhl.inttoptr"(%zero) : (i32) -> !udonhl.ptr<0>
    %buffer = func.call @clCreateBuffer(%context, %flags, %bytes, %null, %null) : (!udonhl.ptr<0>, i64, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> !udonhl.ptr<0>
    %address = udonhl.address_of @image : !udonhl.ptr<0>
    "udonhl.store"(%buffer, %address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    return
  }
}

// -----

module {
  // expected-error@+1 {{UDONCL_PUBLIC variable image is a CL_MEM_READ_WRITE buffer; public read-write buffers are not supported yet}}
  "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, initial_value = 0 : i32, linkage = "internal", storage_type = !udonhl.ptr<0>, sym_name = "image"}> {udoncl.public = "float32"} : () -> ()
  func.func private @clCreateBuffer(!udonhl.ptr<0>, i64, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> !udonhl.ptr<0>
  func.func @Initialize(%context: !udonhl.ptr<0>) attributes {udoncl.export} {
    %flags = arith.constant 1 : i64
    %bytes = arith.constant 64 : i32
    %zero = arith.constant 0 : i32
    %null = "udonhl.inttoptr"(%zero) : (i32) -> !udonhl.ptr<0>
    %buffer = func.call @clCreateBuffer(%context, %flags, %bytes, %null, %null) : (!udonhl.ptr<0>, i64, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> !udonhl.ptr<0>
    %address = udonhl.address_of @image : !udonhl.ptr<0>
    "udonhl.store"(%buffer, %address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    return
  }
}

// -----

module {
  "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, initial_value = 0 : i32, linkage = "internal", storage_type = !udonhl.ptr<0>, sym_name = "image"}> {udoncl.public = "float32"} : () -> ()
  // expected-error@+1 {{UDONCL_PUBLIC variables image and alias hold the same Device buffer}}
  "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, initial_value = 0 : i32, linkage = "internal", storage_type = !udonhl.ptr<0>, sym_name = "alias"}> {udoncl.public = "float32"} : () -> ()
  func.func private @clCreateBuffer(!udonhl.ptr<0>, i64, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> !udonhl.ptr<0>
  func.func @Initialize(%context: !udonhl.ptr<0>) attributes {udoncl.export} {
    %flags = arith.constant 4 : i64
    %bytes = arith.constant 64 : i32
    %zero = arith.constant 0 : i32
    %null = "udonhl.inttoptr"(%zero) : (i32) -> !udonhl.ptr<0>
    %buffer = func.call @clCreateBuffer(%context, %flags, %bytes, %null, %null) : (!udonhl.ptr<0>, i64, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> !udonhl.ptr<0>
    %address = udonhl.address_of @image : !udonhl.ptr<0>
    "udonhl.store"(%buffer, %address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    %alias = udonhl.address_of @alias : !udonhl.ptr<0>
    "udonhl.store"(%buffer, %alias) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    return
  }
}
