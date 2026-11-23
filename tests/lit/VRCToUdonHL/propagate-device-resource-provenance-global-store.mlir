// RUN: %vrc-opt --vrc-discover-static-device-resources --vrc-propagate-device-resource-provenance %s | %FileCheck %s

// A load of a Host global cell reads the handle a store of the same
// invocation left in it on every path to the load, even across calls; a
// load that a path without such a store reaches reads every handle any write
// left in the cell, including its zero initial value.

// CHECK: vrc.static_device_global_cell_provenance = {cells = [{global = @matrix, may_be_null = true, resource_ids = [0 : i32], unknown = false}]
// CHECK-LABEL: func.func @AUpload
// CHECK: call @clEnqueueWriteBuffer
// CHECK-SAME: may_be_null = false, resource_ids = [0 : i32]
// CHECK: call @clEnqueueReadBuffer
// CHECK-SAME: may_be_null = true, resource_ids = [0 : i32]
// CHECK-LABEL: func.func @BRelease
// CHECK: call @clReleaseMemObject
// CHECK-SAME: may_be_null = true, resource_ids = [0 : i32]
module {
  "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, initial_value = 0 : i32, linkage = "internal", storage_type = !udonhl.ptr<0>, sym_name = "matrix"}> : () -> ()
  func.func private @clCreateBuffer() -> !udonhl.ptr<0>
  func.func private @clEnqueueWriteBuffer(!udonhl.ptr<0>) -> i32
  func.func private @clEnqueueReadBuffer(!udonhl.ptr<0>) -> i32
  func.func private @clReleaseMemObject(!udonhl.ptr<0>) -> i32
  func.func private @unrelated() -> i32

  func.func @AUpload(%skip: i1) -> i32 attributes {udoncl.export} {
    %resource = func.call @clCreateBuffer() : () -> !udonhl.ptr<0>
    %address = udonhl.address_of @matrix : !udonhl.ptr<0>
    "udonhl.store"(%resource, %address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    %ignored = func.call @unrelated() : () -> i32
    %loaded = "udonhl.load"(%address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    %written = func.call @clEnqueueWriteBuffer(%loaded) : (!udonhl.ptr<0>) -> i32
    cf.cond_br %skip, ^join, ^store
  ^store:
    %zero = arith.constant 0 : i32
    %null = "udonhl.inttoptr"(%zero) : (i32) -> !udonhl.ptr<0>
    "udonhl.store"(%null, %address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    cf.br ^join
  ^join:
    %reloaded = "udonhl.load"(%address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    %read = func.call @clEnqueueReadBuffer(%reloaded) : (!udonhl.ptr<0>) -> i32
    return %read : i32
  }

  func.func @BRelease() -> i32 attributes {udoncl.export} {
    %address = udonhl.address_of @matrix : !udonhl.ptr<0>
    %loaded = "udonhl.load"(%address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    %status = func.call @clReleaseMemObject(%loaded) : (!udonhl.ptr<0>) -> i32
    return %status : i32
  }
}
