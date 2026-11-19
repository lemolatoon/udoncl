// RUN: %vrc-opt --vrc-discover-static-device-resources --vrc-propagate-device-resource-provenance %s | %FileCheck %s

// A status from a call without memory effects joins an effectful status. The
// branch that routed it to the join already excluded success, so the joined
// success edge must only see the effectful call's success case, where the
// out-parameter holds the resolved Device buffer.

// CHECK-LABEL: func.func @Run
// CHECK: call @clReleaseMemObject
// CHECK-SAME: may_be_null = false
// CHECK-SAME: resource_ids = [0 : i32]
// CHECK-SAME: unknown = false
module {
  "udonhl.global"() <{address_space = 0 : i32, alignment = 8 : i64, linkage = "internal", storage_type = !llvm.array<88 x i8>, sym_name = "bindings"}> {vrc.buffer_view_layout = {count = 1 : i32, offsets = array<i32: 0>, schema_version = 1 : i32, stride_bytes = 88 : i32}} : () -> ()
  func.func private @clCreateBuffer() -> !udonhl.ptr<0>
  func.func private @clReleaseMemObject(!udonhl.ptr<0>) -> i32
  func.func private @udonclDeviceBufferAdopt(!udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>) -> i32
  func.func private @udonclDeviceBufferView(!udonhl.ptr<0>, i32, i32, i32, !udonhl.ptr<0>) -> i32
  func.func private @udonclBufferViewResolve(!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
  func.func private @PriorStatus() -> i32

  func.func private @ResolveChecked(%expected: i32, %memory: !udonhl.ptr<0>, %offset: !udonhl.ptr<0>) -> i32 {
    %zero = arith.constant 0 : i32
    %two = arith.constant 2 : i32
    %four = arith.constant 4 : i32
    %false = arith.constant false
    %null = "udonhl.inttoptr"(%zero) : (i32) -> !udonhl.ptr<0>
    %length = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    "udonhl.store"(%zero, %length) <{alignment = 4 : i64}> : (i32, !udonhl.ptr<0>) -> ()
    %bindings = udonhl.address_of @bindings : !udonhl.ptr<0>
    %status = func.call @udonclBufferViewResolve(%bindings, %memory, %offset, %length) : (!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    %ok = arith.cmpi eq, %status, %zero : i32
    cf.cond_br %ok, ^check, ^exit(%status : i32)
  ^check:
    %resolved = "udonhl.load"(%memory) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    %nullint = "udonhl.ptrtoint"(%null) : (!udonhl.ptr<0>) -> i32
    %resolvedint = "udonhl.ptrtoint"(%resolved) : (!udonhl.ptr<0>) -> i32
    %nonnull = arith.cmpi ne, %resolvedint, %nullint : i32
    %actual = "udonhl.load"(%length) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> i32
    %matches = arith.cmpi eq, %actual, %expected : i32
    %valid = arith.select %nonnull, %matches, %false : i1
    %checked = arith.select %valid, %zero, %two : i32
    cf.br ^exit(%checked : i32)
  ^exit(%result: i32):
    return %result : i32
  }

  func.func @Run() -> i32 attributes {udoncl.export, vrc.export_id = 1 : i32} {
    %zero = arith.constant 0 : i32
    %four = arith.constant 4 : i32
    %five = arith.constant 5 : i32
    %eight = arith.constant 8 : i32
    %elements = arith.constant 16 : i32
    %dtype = arith.constant 6 : i32
    %bytes = arith.constant 64 : i32
    %context = "udonhl.inttoptr"(%zero) : (i32) -> !udonhl.ptr<0>
    %resource = func.call @clCreateBuffer() : () -> !udonhl.ptr<0>
    %handle = "udonhl.alloca"(%eight) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %adopted = func.call @udonclDeviceBufferAdopt(%context, %resource, %bytes, %handle) : (!udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>) -> i32
    %bindings = udonhl.address_of @bindings : !udonhl.ptr<0>
    %viewed = func.call @udonclDeviceBufferView(%handle, %zero, %elements, %dtype, %bindings) : (!udonhl.ptr<0>, i32, i32, i32, !udonhl.ptr<0>) -> i32
    %memory = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %offset = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    "udonhl.store"(%context, %memory) <{alignment = 4 : i64}> : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    "udonhl.store"(%zero, %offset) <{alignment = 4 : i64}> : (i32, !udonhl.ptr<0>) -> ()
    %prior = func.call @PriorStatus() : () -> i32
    %prior_ok = arith.cmpi eq, %prior, %zero : i32
    cf.cond_br %prior_ok, ^resolve, ^merge(%prior : i32)
  ^resolve:
    %status = func.call @ResolveChecked(%bytes, %memory, %offset) : (i32, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    cf.br ^merge(%status : i32)
  ^merge(%merged: i32):
    %ok = arith.cmpi eq, %merged, %zero : i32
    cf.cond_br %ok, ^use, ^done(%merged : i32)
  ^use:
    %loaded = "udonhl.load"(%memory) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    %released = func.call @clReleaseMemObject(%loaded) : (!udonhl.ptr<0>) -> i32
    cf.br ^done(%released : i32)
  ^done(%result: i32):
    return %result : i32
  }
}
