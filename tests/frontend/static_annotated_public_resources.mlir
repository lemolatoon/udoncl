module {
  "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, initial_value = 0 : i32, linkage = "internal", storage_type = !udonhl.ptr<0>, sym_name = "x"}> {udoncl.public = "float32"} : () -> ()
  "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, initial_value = 0 : i32, linkage = "internal", storage_type = !udonhl.ptr<0>, sym_name = "result"}> {udoncl.public = "int32"} : () -> ()
  "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, initial_value = 0 : i32, linkage = "internal", storage_type = !udonhl.ptr<0>, sym_name = "y"}> {udoncl.public = "float32"} : () -> ()
  "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, initial_value = 0 : i32, linkage = "internal", storage_type = !udonhl.ptr<0>, sym_name = "scratch"}> : () -> ()
  func.func private @clCreateBuffer(!udonhl.ptr<0>, i64, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> !udonhl.ptr<0>
  func.func private @clReleaseMemObject(!udonhl.ptr<0>) -> i32

  // A helper called once per buffer: each call is its own allocation
  // context once contexts are specialized.
  func.func @create(%context: !udonhl.ptr<0>, %flags: i64, %bytes: i32) -> !udonhl.ptr<0> {
    %zero = arith.constant 0 : i32
    %null = "udonhl.inttoptr"(%zero) : (i32) -> !udonhl.ptr<0>
    %buffer = func.call @clCreateBuffer(%context, %flags, %bytes, %null, %null) : (!udonhl.ptr<0>, i64, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> !udonhl.ptr<0>
    return %buffer : !udonhl.ptr<0>
  }

  func.func @Initialize(%context: !udonhl.ptr<0>) attributes {udoncl.export} {
    %read_only = arith.constant 4 : i64
    %write_only = arith.constant 2 : i64
    %read_write = arith.constant 1 : i64
    %bytes = arith.constant 128 : i32
    %scratch_bytes = arith.constant 64 : i32
    %x = func.call @create(%context, %read_only, %bytes) : (!udonhl.ptr<0>, i64, i32) -> !udonhl.ptr<0>
    %x_address = udonhl.address_of @x : !udonhl.ptr<0>
    "udonhl.store"(%x, %x_address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    %y = func.call @create(%context, %read_only, %bytes) : (!udonhl.ptr<0>, i64, i32) -> !udonhl.ptr<0>
    %y_address = udonhl.address_of @y : !udonhl.ptr<0>
    "udonhl.store"(%y, %y_address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    %result = func.call @create(%context, %write_only, %bytes) : (!udonhl.ptr<0>, i64, i32) -> !udonhl.ptr<0>
    %result_address = udonhl.address_of @result : !udonhl.ptr<0>
    "udonhl.store"(%result, %result_address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    // A private buffer stays private, whatever its flags.
    %scratch = func.call @create(%context, %read_write, %scratch_bytes) : (!udonhl.ptr<0>, i64, i32) -> !udonhl.ptr<0>
    %scratch_address = udonhl.address_of @scratch : !udonhl.ptr<0>
    "udonhl.store"(%scratch, %scratch_address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    return
  }

  // Disposing a buffer leaves the variable null, which it may be.
  func.func @Dispose() attributes {udoncl.export} {
    %address = udonhl.address_of @x : !udonhl.ptr<0>
    %buffer = "udonhl.load"(%address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    %status = func.call @clReleaseMemObject(%buffer) : (!udonhl.ptr<0>) -> i32
    %zero = arith.constant 0 : i32
    %null = "udonhl.inttoptr"(%zero) : (i32) -> !udonhl.ptr<0>
    "udonhl.store"(%null, %address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    return
  }
}
