module {
  func.func private @clCreateBuffer() -> !udonhl.ptr<0>
  func.func private @clReleaseMemObject(!udonhl.ptr<0>) -> i32

  func.func @through_cell(%buffer: !udonhl.ptr<0>) -> !udonhl.ptr<0> {
    %four = arith.constant 4 : i32
    %cell = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    "udonhl.store"(%buffer, %cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    %loaded = "udonhl.load"(%cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    return %loaded : !udonhl.ptr<0>
  }

  func.func @passthrough(%buffer: !udonhl.ptr<0>) -> !udonhl.ptr<0> {
    return %buffer : !udonhl.ptr<0>
  }

  func.func @Run(%condition: i1) attributes {udoncl.export} {
    %four = arith.constant 4 : i32
    %zero = arith.constant 0 : i32
    %null = "udonhl.inttoptr"(%zero) : (i32) -> !udonhl.ptr<0>
    %first = func.call @clCreateBuffer() : () -> !udonhl.ptr<0>
    %second = func.call @clCreateBuffer() : () -> !udonhl.ptr<0>

    %first_alias = func.call @through_cell(%first) : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    %second_alias = func.call @through_cell(%second) : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    %status0 = func.call @clReleaseMemObject(%first_alias) : (!udonhl.ptr<0>) -> i32
    %status1 = func.call @clReleaseMemObject(%second_alias) : (!udonhl.ptr<0>) -> i32

    %overwrite = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    "udonhl.store"(%first, %overwrite) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    "udonhl.store"(%second, %overwrite) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    %overwritten = "udonhl.load"(%overwrite) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    %status2 = func.call @clReleaseMemObject(%overwritten) : (!udonhl.ptr<0>) -> i32

    %joined_cell = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    cf.cond_br %condition, ^resource, ^other
  ^resource:
    "udonhl.store"(%first, %joined_cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    cf.br ^merge
  ^other:
    "udonhl.store"(%second, %joined_cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    cf.br ^merge
  ^merge:
    %joined = "udonhl.load"(%joined_cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    %status3 = func.call @clReleaseMemObject(%joined) : (!udonhl.ptr<0>) -> i32

    %nullable_cell = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    cf.cond_br %condition, ^nonnull, ^null
  ^nonnull:
    "udonhl.store"(%first, %nullable_cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    cf.br ^nullable_merge
  ^null:
    "udonhl.store"(%null, %nullable_cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    cf.br ^nullable_merge
  ^nullable_merge:
    %nullable = "udonhl.load"(%nullable_cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    %status4 = func.call @clReleaseMemObject(%nullable) : (!udonhl.ptr<0>) -> i32

    %cascade_cell = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    cf.cond_br %condition, ^cascade_store_first, ^cascade_missing
  ^cascade_store_first:
    "udonhl.store"(%first, %cascade_cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    cf.br ^cascade_merge_one
  ^cascade_missing:
    cf.br ^cascade_merge_one
  ^cascade_merge_one:
    cf.cond_br %condition, ^cascade_keep, ^cascade_store_second
  ^cascade_keep:
    cf.br ^cascade_merge_two
  ^cascade_store_second:
    "udonhl.store"(%second, %cascade_cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    cf.br ^cascade_merge_two
  ^cascade_merge_two:
    %cascade = "udonhl.load"(%cascade_cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    %dead = func.call @passthrough(%cascade) : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    return
  }

  func.func @RunLoop(%condition: i1) attributes {udoncl.export} {
    %four = arith.constant 4 : i32
    %first = func.call @clCreateBuffer() : () -> !udonhl.ptr<0>
    %second = func.call @clCreateBuffer() : () -> !udonhl.ptr<0>
    %cell = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    "udonhl.store"(%first, %cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    cf.br ^loop
  ^loop:
    cf.cond_br %condition, ^backedge, ^exit
  ^backedge:
    "udonhl.store"(%second, %cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    cf.br ^loop
  ^exit:
    %loaded = "udonhl.load"(%cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    %status = func.call @clReleaseMemObject(%loaded) : (!udonhl.ptr<0>) -> i32
    return
  }

  func.func @RunOrdering(%condition: i1) attributes {udoncl.export} {
    %four = arith.constant 4 : i32
    %seven = arith.constant 7 : i32
    %unknown = "udonhl.inttoptr"(%seven) : (i32) -> !udonhl.ptr<0>
    %resource = func.call @clCreateBuffer() : () -> !udonhl.ptr<0>
    %cell = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    cf.cond_br %condition, ^store_resource, ^store_unknown
  ^store_resource:
    "udonhl.store"(%resource, %cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    cf.br ^merge
  ^store_unknown:
    "udonhl.store"(%unknown, %cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    cf.br ^merge
  ^merge:
    %loaded = "udonhl.load"(%cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    %unused = func.call @passthrough(%loaded) : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    return
  }
}
