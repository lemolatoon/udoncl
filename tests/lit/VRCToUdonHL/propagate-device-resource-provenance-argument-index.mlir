// RUN: %vrc-opt --vrc-discover-static-device-resources --vrc-propagate-device-resource-provenance --split-input-file --verify-diagnostics %s | %FileCheck %s

// A private helper indexes a strided view array with a function argument. The
// index is not a loop induction inside the helper, so every in-bounds element
// of the array is a candidate and each one must hold an initialized view.

// CHECK-LABEL: func.func private @ResolveElement
// CHECK: call @udonclBufferViewResolve
// CHECK-SAME: resource_ids = [0 : i32]
// CHECK-SAME: unknown = false
module {
  func.func private @clCreateBuffer() -> !udonhl.ptr<0>
  func.func private @udonclDeviceBufferAdopt(!udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>) -> i32
  func.func private @udonclDeviceBufferView(!udonhl.ptr<0>, i32, i32, i32, !udonhl.ptr<0>) -> i32
  func.func private @udonclBufferViewResolve(!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32

  func.func private @ResolveElement(%views: !udonhl.ptr<0>, %index: i32, %raw: !udonhl.ptr<0>, %offset: !udonhl.ptr<0>, %length: !udonhl.ptr<0>) -> i32 {
    %eighty_eight = arith.constant 88 : i32
    %byte_offset = arith.muli %index, %eighty_eight : i32
    %current = "udonhl.ptr_add"(%views, %byte_offset) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    %resolved = func.call @udonclBufferViewResolve(%current, %raw, %offset, %length) : (!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    return %resolved : i32
  }

  func.func @ResolveEveryElement() attributes {udoncl.export, vrc.export_id = 1 : i32} {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %two = arith.constant 2 : i32
    %four = arith.constant 4 : i32
    %eight = arith.constant 8 : i32
    %eighty_eight = arith.constant 88 : i32
    %one_hundred_seventy_six = arith.constant 176 : i32
    %elements = arith.constant 16 : i32
    %dtype = arith.constant 6 : i32
    %bytes = arith.constant 64 : i32
    %context = "udonhl.inttoptr"(%zero) : (i32) -> !udonhl.ptr<0>
    %resource = func.call @clCreateBuffer() : () -> !udonhl.ptr<0>
    %handle = "udonhl.alloca"(%eight) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %adopted = func.call @udonclDeviceBufferAdopt(%context, %resource, %bytes, %handle) : (!udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>) -> i32
    %views = "udonhl.alloca"(%one_hundred_seventy_six) <{alignment = 8 : i64}> : (i32) -> !udonhl.ptr<0>
    %first_viewed = func.call @udonclDeviceBufferView(%handle, %zero, %elements, %dtype, %views) : (!udonhl.ptr<0>, i32, i32, i32, !udonhl.ptr<0>) -> i32
    %second = "udonhl.ptr_add"(%views, %eighty_eight) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    %second_viewed = func.call @udonclDeviceBufferView(%handle, %zero, %elements, %dtype, %second) : (!udonhl.ptr<0>, i32, i32, i32, !udonhl.ptr<0>) -> i32
    %raw = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %offset = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %length = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    cf.br ^loop(%zero : i32)
  ^loop(%index: i32):
    %resolved = func.call @ResolveElement(%views, %index, %raw, %offset, %length) : (!udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    %next = arith.addi %index, %one overflow<nuw> : i32
    %continue = arith.cmpi ult, %next, %two : i32
    cf.cond_br %continue, ^loop(%next : i32), ^done
  ^done:
    return
  }
}

// -----

// The same helper must be rejected when one in-bounds element of the array is
// never initialized and the index is supplied by the export's caller.
module {
  func.func private @clCreateBuffer() -> !udonhl.ptr<0>
  func.func private @udonclDeviceBufferAdopt(!udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>) -> i32
  func.func private @udonclDeviceBufferView(!udonhl.ptr<0>, i32, i32, i32, !udonhl.ptr<0>) -> i32
  func.func private @udonclBufferViewResolve(!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32

  func.func private @ResolveElement(%views: !udonhl.ptr<0>, %index: i32, %raw: !udonhl.ptr<0>, %offset: !udonhl.ptr<0>, %length: !udonhl.ptr<0>) -> i32 {
    %eighty_eight = arith.constant 88 : i32
    %byte_offset = arith.muli %index, %eighty_eight : i32
    %current = "udonhl.ptr_add"(%views, %byte_offset) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    // expected-error @below {{udonclBufferViewResolve requires an initialized view subobject in @ResolveElement from root @RejectPartiallyInitialized}}
    %resolved = func.call @udonclBufferViewResolve(%current, %raw, %offset, %length) : (!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    return %resolved : i32
  }

  func.func @RejectPartiallyInitialized(%index: i32) attributes {udoncl.export, vrc.export_id = 1 : i32} {
    %zero = arith.constant 0 : i32
    %four = arith.constant 4 : i32
    %eight = arith.constant 8 : i32
    %one_hundred_seventy_six = arith.constant 176 : i32
    %elements = arith.constant 16 : i32
    %dtype = arith.constant 6 : i32
    %bytes = arith.constant 64 : i32
    %context = "udonhl.inttoptr"(%zero) : (i32) -> !udonhl.ptr<0>
    %resource = func.call @clCreateBuffer() : () -> !udonhl.ptr<0>
    %handle = "udonhl.alloca"(%eight) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %adopted = func.call @udonclDeviceBufferAdopt(%context, %resource, %bytes, %handle) : (!udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>) -> i32
    %views = "udonhl.alloca"(%one_hundred_seventy_six) <{alignment = 8 : i64}> : (i32) -> !udonhl.ptr<0>
    %first_viewed = func.call @udonclDeviceBufferView(%handle, %zero, %elements, %dtype, %views) : (!udonhl.ptr<0>, i32, i32, i32, !udonhl.ptr<0>) -> i32
    %raw = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %offset = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %length = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %resolved = func.call @ResolveElement(%views, %index, %raw, %offset, %length) : (!udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    return
  }
}

// -----

// A declared buffer-view layout bounds the selectable elements of storage
// that reserves a tail beyond its last element.

// CHECK-LABEL: func.func private @ResolveBinding
// CHECK: call @udonclBufferViewResolve
// CHECK-SAME: resource_ids = [0 : i32]
// CHECK-SAME: unknown = false
module {
  "udonhl.global"() <{address_space = 0 : i32, alignment = 8 : i64, linkage = "internal", storage_type = !llvm.array<256 x i8>, sym_name = "bindings"}> {vrc.buffer_view_layout = {count = 2 : i32, offsets = array<i32: 0>, schema_version = 1 : i32, stride_bytes = 88 : i32}} : () -> ()
  func.func private @clCreateBuffer() -> !udonhl.ptr<0>
  func.func private @udonclDeviceBufferAdopt(!udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>) -> i32
  func.func private @udonclDeviceBufferView(!udonhl.ptr<0>, i32, i32, i32, !udonhl.ptr<0>) -> i32
  func.func private @udonclBufferViewResolve(!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32

  func.func private @ResolveBinding(%index: i32, %raw: !udonhl.ptr<0>, %offset: !udonhl.ptr<0>, %length: !udonhl.ptr<0>) -> i32 {
    %eighty_eight = arith.constant 88 : i32
    %bindings = udonhl.address_of @bindings : !udonhl.ptr<0>
    %byte_offset = arith.muli %index, %eighty_eight : i32
    %current = "udonhl.ptr_add"(%bindings, %byte_offset) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    %resolved = func.call @udonclBufferViewResolve(%current, %raw, %offset, %length) : (!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    return %resolved : i32
  }

  func.func @ResolveDeclaredElement(%index: i32) attributes {udoncl.export, vrc.export_id = 1 : i32} {
    %zero = arith.constant 0 : i32
    %four = arith.constant 4 : i32
    %eight = arith.constant 8 : i32
    %eighty_eight = arith.constant 88 : i32
    %elements = arith.constant 16 : i32
    %dtype = arith.constant 6 : i32
    %bytes = arith.constant 64 : i32
    %context = "udonhl.inttoptr"(%zero) : (i32) -> !udonhl.ptr<0>
    %resource = func.call @clCreateBuffer() : () -> !udonhl.ptr<0>
    %handle = "udonhl.alloca"(%eight) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %adopted = func.call @udonclDeviceBufferAdopt(%context, %resource, %bytes, %handle) : (!udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>) -> i32
    %bindings = udonhl.address_of @bindings : !udonhl.ptr<0>
    %first = func.call @udonclDeviceBufferView(%handle, %zero, %elements, %dtype, %bindings) : (!udonhl.ptr<0>, i32, i32, i32, !udonhl.ptr<0>) -> i32
    %second_view = "udonhl.ptr_add"(%bindings, %eighty_eight) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    %second = func.call @udonclDeviceBufferView(%handle, %zero, %elements, %dtype, %second_view) : (!udonhl.ptr<0>, i32, i32, i32, !udonhl.ptr<0>) -> i32
    %raw = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %offset = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %length = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %resolved = func.call @ResolveBinding(%index, %raw, %offset, %length) : (i32, !udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    return
  }
}
