module {
  "udonhl.global"() <{address_space = 0 : i32, alignment = 8 : i64, linkage = "internal", storage_type = !llvm.array<40 x i8>, sym_name = "session"}> : () -> ()
  "udonhl.global"() <{address_space = 0 : i32, alignment = 8 : i64, linkage = "internal", storage_type = !llvm.array<88 x i8>, sym_name = "bindings"}> : () -> ()

  func.func private @udononnxCopyBufferView(!udonhl.ptr<0>, !udonhl.ptr<0>)

  func.func @Run(%condition: i1) attributes {udoncl.export} {
    %eight = arith.constant 8 : i32
    %session = udonhl.address_of @session : !udonhl.ptr<0>
    %bindings = udonhl.address_of @bindings : !udonhl.ptr<0>
    %bindings_cell = "udonhl.ptr_add"(%session, %eight) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    cf.cond_br %condition, ^initialize, ^skip
  ^initialize:
    "udonhl.store"(%bindings, %bindings_cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    cf.br ^use
  ^skip:
    cf.br ^use
  ^use:
    %loaded = "udonhl.load"(%bindings_cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    func.call @udononnxCopyBufferView(%loaded, %bindings) : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    return
  }
}
