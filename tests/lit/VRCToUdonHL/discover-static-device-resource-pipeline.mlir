// RUN: %vrc-opt --vrc-discover-static-device-resource-pipeline %s | %FileCheck %s

// The pipeline discovers one allocation context per clCreateBuffer, follows
// it through a global cell into a user kernel launch, and assigns it a
// Unity Texture.
// CHECK: vrc.static_device_global_cell_provenance = {cells = [{global = @image, may_be_null = true, resource_ids = [0 : i32], unknown = false}]
// CHECK-SAME: vrc.static_device_resource_discovery = {contexts = [{allocation_site = {function = @Initialize, occurrence = 0 : i32, primitive = "clCreateBuffer", resource_result = 0 : i32}, call_path = [], context_id = 0 : i32, root = @Initialize}]
// CHECK-SAME: vrc.static_device_resource_inventory =
// CHECK-SAME: vrc.static_public_resource_aliases =
// CHECK-SAME: vrc.unity_texture_assignments = {assignments = [{capacity_bytes = 128 : i64
// CHECK: func.func private @__vrc_enqueue_user_1024_1_fill({{.*}}) -> i32 attributes {vrc.device_resource_operands = [{access = "write", capture = "none", index = 1 : i32}]}
module {
  "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, initial_value = 0 : i32, linkage = "internal", storage_type = !udonhl.ptr<0>, sym_name = "image"}> : () -> ()
  func.func private @clCreateBuffer(!udonhl.ptr<0>, i64, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> !udonhl.ptr<0>
  func.func private @__vrc_enqueue_user_1024_1_fill(!udonhl.ptr<0>, !udonhl.ptr<0>, i32) -> i32
  func.func @Initialize(%context: !udonhl.ptr<0>) attributes {udoncl.export} {
    %flags = arith.constant 1 : i64
    %bytes = arith.constant 128 : i32
    %zero = arith.constant 0 : i32
    %null = "udonhl.inttoptr"(%zero) : (i32) -> !udonhl.ptr<0>
    %buffer = func.call @clCreateBuffer(%context, %flags, %bytes, %null, %null) : (!udonhl.ptr<0>, i64, i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> !udonhl.ptr<0>
    %address = udonhl.address_of @image : !udonhl.ptr<0>
    "udonhl.store"(%buffer, %address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    return
  }
  func.func @Run(%queue: !udonhl.ptr<0>) -> i32 attributes {udoncl.export} {
    %address = udonhl.address_of @image : !udonhl.ptr<0>
    %buffer = "udonhl.load"(%address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    %value = arith.constant 7 : i32
    %status = func.call @__vrc_enqueue_user_1024_1_fill(%queue, %buffer, %value) : (!udonhl.ptr<0>, !udonhl.ptr<0>, i32) -> i32
    return %status : i32
  }
}
