module {
  "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, initial_value = array<i8: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>, linkage = "internal", storage_type = !llvm.array<20 x i8>, sym_name = "persistent_view"}> : () -> ()
  func.func private @clCreateBuffer() -> !udonhl.ptr<0>
  func.func private @clReleaseMemObject(!udonhl.ptr<0>) -> i32
  func.func private @udonclDeviceBufferAdopt(!udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>) -> i32
  func.func private @udonclDeviceBufferView(!udonhl.ptr<0>, i32, i32, i32, !udonhl.ptr<0>) -> i32
  func.func private @udonclBufferViewResolve(!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
  func.func private @validateView(!udonhl.ptr<0>) -> i32

  func.func @ConditionalGlobalInitialize(%initialize: i1) attributes {udoncl.export, vrc.export_id = 2 : i32} {
    cf.cond_br %initialize, ^bb1, ^bb2
  ^bb1:
    %zero = arith.constant 0 : i32
    %eight = arith.constant 8 : i32
    %elements = arith.constant 16 : i32
    %dtype = arith.constant 6 : i32
    %bytes = arith.constant 64 : i32
    %context = "udonhl.inttoptr"(%zero) : (i32) -> !udonhl.ptr<0>
    %resource = func.call @clCreateBuffer() : () -> !udonhl.ptr<0>
    %handle = "udonhl.alloca"(%eight) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %adopted = func.call @udonclDeviceBufferAdopt(%context, %resource, %bytes, %handle) : (!udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>) -> i32
    %view = udonhl.address_of @persistent_view : !udonhl.ptr<0>
    %viewed = func.call @udonclDeviceBufferView(%handle, %zero, %elements, %dtype, %view) : (!udonhl.ptr<0>, i32, i32, i32, !udonhl.ptr<0>) -> i32
    cf.br ^bb2
  ^bb2:
    return
  }

  func.func @ConsumeGlobalView() attributes {udoncl.export, vrc.export_id = 3 : i32} {
    %four = arith.constant 4 : i32
    %view = udonhl.address_of @persistent_view : !udonhl.ptr<0>
    %raw = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %offset = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %length = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %resolved = func.call @udonclBufferViewResolve(%view, %raw, %offset, %length) : (!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    return
  }

  func.func private @udononnxCopyBufferView(%destination: !udonhl.ptr<0>, %source: !udonhl.ptr<0>) {
    %four = arith.constant 4 : i32
    %eight = arith.constant 8 : i32
    %twelve = arith.constant 12 : i32
    %sixteen = arith.constant 16 : i32
    %source4 = "udonhl.ptr_add"(%source, %four) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    %destination4 = "udonhl.ptr_add"(%destination, %four) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    %source8 = "udonhl.ptr_add"(%source, %eight) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    %destination8 = "udonhl.ptr_add"(%destination, %eight) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    %source12 = "udonhl.ptr_add"(%source, %twelve) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    %destination12 = "udonhl.ptr_add"(%destination, %twelve) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    %source16 = "udonhl.ptr_add"(%source, %sixteen) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    %destination16 = "udonhl.ptr_add"(%destination, %sixteen) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    return
  }

  func.func private @initializeDirectView(%slot: i32, %generation: i32, %view: !udonhl.ptr<0>) {
    %zero = arith.constant 0 : i32
    %four = arith.constant 4 : i32
    %eight = arith.constant 8 : i32
    %twelve = arith.constant 12 : i32
    %sixteen = arith.constant 16 : i32
    %bytes = arith.constant 64 : i32
    %dtype = arith.constant 6 : i32
    "udonhl.store"(%slot, %view) <{alignment = 4 : i64}> : (i32, !udonhl.ptr<0>) -> ()
    %view4 = "udonhl.ptr_add"(%view, %four) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    "udonhl.store"(%generation, %view4) <{alignment = 4 : i64}> : (i32, !udonhl.ptr<0>) -> ()
    %view8 = "udonhl.ptr_add"(%view, %eight) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    "udonhl.store"(%zero, %view8) <{alignment = 4 : i64}> : (i32, !udonhl.ptr<0>) -> ()
    %view12 = "udonhl.ptr_add"(%view, %twelve) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    "udonhl.store"(%bytes, %view12) <{alignment = 4 : i64}> : (i32, !udonhl.ptr<0>) -> ()
    %view16 = "udonhl.ptr_add"(%view, %sixteen) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    "udonhl.store"(%dtype, %view16) <{alignment = 4 : i64}> : (i32, !udonhl.ptr<0>) -> ()
    return
  }

  func.func private @initializeDirectViewBranch(%condition: i1, %slot: i32, %generation: i32, %view: !udonhl.ptr<0>) {
    cf.cond_br %condition, ^bb1, ^bb2
  ^bb1:
    func.call @initializeDirectView(%slot, %generation, %view) : (i32, i32, !udonhl.ptr<0>) -> ()
    cf.br ^bb3
  ^bb2:
    func.call @initializeDirectView(%slot, %generation, %view) : (i32, i32, !udonhl.ptr<0>) -> ()
    cf.br ^bb3
  ^bb3:
    return
  }

  func.func private @createConditionalView(%fail: i1, %destination: !udonhl.ptr<0>, %source: !udonhl.ptr<0>) -> i32 {
    %failure = arith.constant 1 : i32
    cf.cond_br %fail, ^bb1, ^bb2
  ^bb1:
    return %failure : i32
  ^bb2:
    func.call @udononnxCopyBufferView(%destination, %source) : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    %status = func.call @validateView(%destination) : (!udonhl.ptr<0>) -> i32
    return %status : i32
  }

  func.func private @forwardConditionalView(%fail: i1, %destination: !udonhl.ptr<0>, %source: !udonhl.ptr<0>) -> i32 {
    %zero = arith.constant 0 : i32
    %status = func.call @createConditionalView(%fail, %destination, %source) : (i1, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    cf.br ^bb0(%status : i32)
  ^bb0(%forwarded_status: i32):
    %succeeded = arith.cmpi eq, %forwarded_status, %zero : i32
    cf.cond_br %succeeded, ^bb1, ^bb2
  ^bb1:
    cf.br ^bb3(%zero : i32)
  ^bb2:
    // Model cleanup between the call and the joined return. The failure edge
    // still proves that the original status cannot be zero.
    %cleanup = arith.addi %forwarded_status, %zero : i32
    cf.br ^bb3(%forwarded_status : i32)
  ^bb3(%forwarded: i32):
    return %forwarded : i32
  }

  func.func @Run(%fail: i1) attributes {udoncl.export, vrc.export_id = 1 : i32} {
    %zero = arith.constant 0 : i32
    %four = arith.constant 4 : i32
    %eight = arith.constant 8 : i32
    %twenty = arith.constant 20 : i32
    %eighty_eight = arith.constant 88 : i32
    %elements = arith.constant 16 : i32
    %dtype = arith.constant 6 : i32
    %bytes = arith.constant 64 : i32
    %context = "udonhl.inttoptr"(%zero) : (i32) -> !udonhl.ptr<0>
    %resource = func.call @clCreateBuffer() : () -> !udonhl.ptr<0>
    %handle = "udonhl.alloca"(%eight) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %adopted = func.call @udonclDeviceBufferAdopt(%context, %resource, %bytes, %handle) : (!udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>) -> i32
    %view = "udonhl.alloca"(%twenty) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %viewed = func.call @udonclDeviceBufferView(%handle, %zero, %elements, %dtype, %view) : (!udonhl.ptr<0>, i32, i32, i32, !udonhl.ptr<0>) -> i32
    %tensor = "udonhl.alloca"(%eighty_eight) <{alignment = 8 : i64}> : (i32) -> !udonhl.ptr<0>
    func.call @udononnxCopyBufferView(%tensor, %view) : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    %copied_tensor = "udonhl.alloca"(%eighty_eight) <{alignment = 8 : i64}> : (i32) -> !udonhl.ptr<0>
    "udonhl.memcpy"(%copied_tensor, %tensor, %eighty_eight) : (!udonhl.ptr<0>, !udonhl.ptr<0>, i32) -> ()
    %slot = "udonhl.load"(%handle) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> i32
    %handle4 = "udonhl.ptr_add"(%handle, %four) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    %generation = "udonhl.load"(%handle4) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> i32
    %direct_view = "udonhl.alloca"(%twenty) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %condition = arith.cmpi ne, %slot, %zero : i32
    func.call @initializeDirectViewBranch(%condition, %slot, %generation, %direct_view) : (i1, i32, i32, !udonhl.ptr<0>) -> ()
    %direct_tensor = "udonhl.alloca"(%eighty_eight) <{alignment = 8 : i64}> : (i32) -> !udonhl.ptr<0>
    %raw_cell = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %null = "udonhl.inttoptr"(%zero) : (i32) -> !udonhl.ptr<0>
    "udonhl.store"(%null, %raw_cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    %offset_cell = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %length_cell = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %create_status = func.call @forwardConditionalView(%fail, %direct_tensor, %direct_view) : (i1, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    cf.br ^bb0(%create_status : i32)
  ^bb0(%forwarded_create_status: i32):
    %created = arith.cmpi eq, %forwarded_create_status, %zero : i32
    cf.cond_br %created, ^bb1, ^bb2
  ^bb1:
    %direct_resolved = func.call @udonclBufferViewResolve(%direct_tensor, %raw_cell, %offset_cell, %length_cell) : (!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    cf.br ^bb3
  ^bb2:
    cf.br ^bb3
  ^bb3:
    %copied_resolved = func.call @udonclBufferViewResolve(%copied_tensor, %raw_cell, %offset_cell, %length_cell) : (!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    %resolved = func.call @udonclBufferViewResolve(%tensor, %raw_cell, %offset_cell, %length_cell) : (!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    %raw = "udonhl.load"(%raw_cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    %released = func.call @clReleaseMemObject(%raw) : (!udonhl.ptr<0>) -> i32
    return
  }

  // Unity export selection hides this low-level dynamic binding entry point.
  // It is deliberately not statically resolvable and must not become an
  // analysis root merely because its source definition is annotated.
  func.func @HiddenDynamicBinding(%slot: i32, %generation: i32) attributes {udoncl.export} {
    %zero = arith.constant 0 : i32
    %four = arith.constant 4 : i32
    %eight = arith.constant 8 : i32
    %twelve = arith.constant 12 : i32
    %sixteen = arith.constant 16 : i32
    %twenty = arith.constant 20 : i32
    %eighty_eight = arith.constant 88 : i32
    %bytes = arith.constant 64 : i32
    %dtype = arith.constant 6 : i32
    %view = "udonhl.alloca"(%twenty) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    "udonhl.store"(%slot, %view) <{alignment = 4 : i64}> : (i32, !udonhl.ptr<0>) -> ()
    %view4 = "udonhl.ptr_add"(%view, %four) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    "udonhl.store"(%generation, %view4) <{alignment = 4 : i64}> : (i32, !udonhl.ptr<0>) -> ()
    %view8 = "udonhl.ptr_add"(%view, %eight) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    "udonhl.store"(%zero, %view8) <{alignment = 4 : i64}> : (i32, !udonhl.ptr<0>) -> ()
    %view12 = "udonhl.ptr_add"(%view, %twelve) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    "udonhl.store"(%bytes, %view12) <{alignment = 4 : i64}> : (i32, !udonhl.ptr<0>) -> ()
    %view16 = "udonhl.ptr_add"(%view, %sixteen) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    "udonhl.store"(%dtype, %view16) <{alignment = 4 : i64}> : (i32, !udonhl.ptr<0>) -> ()
    %tensor = "udonhl.alloca"(%eighty_eight) <{alignment = 8 : i64}> : (i32) -> !udonhl.ptr<0>
    func.call @udononnxCopyBufferView(%tensor, %view) : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    return
  }
}
