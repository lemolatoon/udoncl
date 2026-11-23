// RUN: %vrc-opt --vrc-discover-static-device-resources --vrc-propagate-device-resource-provenance %s | %FileCheck %s

// A global whose address a helper receives as an out-parameter, and only
// stores to and loads from, stays a tracked global cell: each call stores the
// buffer it creates into the global the caller names. A global whose address
// reaches an external function is not tracked.

// CHECK: vrc.static_device_global_cell_provenance = {cells = [{global = @image, may_be_null = true, resource_ids = [0 : i32], unknown = false}, {global = @logits, may_be_null = true, resource_ids = [1 : i32], unknown = false}]
// CHECK-LABEL: func.func @Run
// CHECK: call @clEnqueueReadBuffer
// CHECK-SAME: may_be_null = true, resource_ids = [1 : i32]
module {
  "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, initial_value = 0 : i32, linkage = "internal", storage_type = !udonhl.ptr<0>, sym_name = "image"}> : () -> ()
  "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, initial_value = 0 : i32, linkage = "internal", storage_type = !udonhl.ptr<0>, sym_name = "logits"}> : () -> ()
  "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, initial_value = 0 : i32, linkage = "internal", storage_type = !udonhl.ptr<0>, sym_name = "escaped"}> : () -> ()
  func.func private @clCreateBuffer(i32) -> !udonhl.ptr<0>
  func.func private @clEnqueueReadBuffer(!udonhl.ptr<0>) -> i32
  func.func private @observe(!udonhl.ptr<0>)

  func.func private @create_image(%out: !udonhl.ptr<0>) -> i32 {
    %flags = arith.constant 4 : i32
    %buffer = func.call @clCreateBuffer(%flags) : (i32) -> !udonhl.ptr<0>
    "udonhl.store"(%buffer, %out) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    %stored = "udonhl.load"(%out) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    %zero = arith.constant 0 : i32
    return %zero : i32
  }

  func.func private @create_logits(%out: !udonhl.ptr<0>) -> i32 {
    %flags = arith.constant 2 : i32
    %buffer = func.call @clCreateBuffer(%flags) : (i32) -> !udonhl.ptr<0>
    "udonhl.store"(%buffer, %out) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    %zero = arith.constant 0 : i32
    return %zero : i32
  }

  func.func @Initialize() attributes {udoncl.export} {
    %image = udonhl.address_of @image : !udonhl.ptr<0>
    %logits = udonhl.address_of @logits : !udonhl.ptr<0>
    %0 = func.call @create_image(%image) : (!udonhl.ptr<0>) -> i32
    %1 = func.call @create_logits(%logits) : (!udonhl.ptr<0>) -> i32
    %escaped = udonhl.address_of @escaped : !udonhl.ptr<0>
    func.call @observe(%escaped) : (!udonhl.ptr<0>) -> ()
    return
  }

  func.func @Run() -> i32 attributes {udoncl.export} {
    %logits = udonhl.address_of @logits : !udonhl.ptr<0>
    %buffer = "udonhl.load"(%logits) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    %status = func.call @clEnqueueReadBuffer(%buffer) : (!udonhl.ptr<0>) -> i32
    return %status : i32
  }
}
