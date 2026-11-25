// RUN: %vrc-opt --vrc-discover-static-device-resource-pipeline %s | %FileCheck %s

// Two cl_mem arrays filled in one loop: after the pipeline unrolls the loop,
// each element is a global cell holding the buffer of its iteration, and
// a constant element read later resolves to it.
// CHECK: vrc.static_device_global_cell_provenance = {cells = [{global = @keys, may_be_null = true, offset = 0 : i32, resource_ids = [0 : i32], unknown = false}, {global = @keys, may_be_null = true, offset = 4 : i32, resource_ids = [2 : i32], unknown = false}, {global = @values, may_be_null = true, offset = 0 : i32, resource_ids = [1 : i32], unknown = false}, {global = @values, may_be_null = true, offset = 4 : i32, resource_ids = [3 : i32], unknown = false}]
// CHECK-LABEL: func.func @Run
// CHECK: call @__vrc_enqueue_user_1024_1_fill({{.*}}) {vrc.static_device_operand_provenance = {{.*}}resource_ids = [3 : i32]
module {
  "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, initial_value = array<i8: 0, 0, 0, 0, 0, 0, 0, 0>, linkage = "internal", storage_type = !llvm.array<2 x ptr>, sym_name = "keys"}> : () -> ()
  "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, initial_value = array<i8: 0, 0, 0, 0, 0, 0, 0, 0>, linkage = "internal", storage_type = !llvm.array<2 x ptr>, sym_name = "values"}> : () -> ()
  func.func private @clCreateBuffer(!udonhl.ptr<0>, i64, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> !udonhl.ptr<0>
  func.func private @__vrc_enqueue_user_1024_1_fill(!udonhl.ptr<0>, !udonhl.ptr<0>, i32) -> i32
  func.func @Initialize(%context: !udonhl.ptr<0>) attributes {udoncl.export} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c4 = arith.constant 4 : i32
    %flags = arith.constant 1 : i64
    %bytes = arith.constant 64 : i32
    %null = "udonhl.inttoptr"(%c0) : (i32) -> !udonhl.ptr<0>
    %keys = udonhl.address_of @keys : !udonhl.ptr<0>
    %values = udonhl.address_of @values : !udonhl.ptr<0>
    cf.br ^loop(%c0 : i32)
  ^loop(%layer: i32):
    %offset = arith.muli %layer, %c4 : i32
    %key = func.call @clCreateBuffer(%context, %flags, %bytes, %null, %null) : (!udonhl.ptr<0>, i64, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> !udonhl.ptr<0>
    %key_cell = "udonhl.ptr_add"(%keys, %offset) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    "udonhl.store"(%key, %key_cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    %value = func.call @clCreateBuffer(%context, %flags, %bytes, %null, %null) : (!udonhl.ptr<0>, i64, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> !udonhl.ptr<0>
    %value_cell = "udonhl.ptr_add"(%values, %offset) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    "udonhl.store"(%value, %value_cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    %next = arith.addi %layer, %c1 : i32
    %more = arith.cmpi slt, %next, %c2 : i32
    cf.cond_br %more, ^loop(%next : i32), ^done
  ^done:
    return
  }
  func.func @Run(%queue: !udonhl.ptr<0>) -> i32 attributes {udoncl.export} {
    %c4 = arith.constant 4 : i32
    %values = udonhl.address_of @values : !udonhl.ptr<0>
    %cell = "udonhl.ptr_add"(%values, %c4) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    %buffer = "udonhl.load"(%cell) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    %fill = arith.constant 7 : i32
    %status = func.call @__vrc_enqueue_user_1024_1_fill(%queue, %buffer, %fill) : (!udonhl.ptr<0>, !udonhl.ptr<0>, i32) -> i32
    return %status : i32
  }
}
