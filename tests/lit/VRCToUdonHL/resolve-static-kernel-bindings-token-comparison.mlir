// RUN: %vrc-opt --split-input-file --vrc-resolve-static-kernel-bindings %s | %FileCheck %s

// A comparison of two physical buffer tokens that dispatches use and that
// resolve to allocations folds: equal exactly when both are the same
// allocation. The blocks only the folded branch reached are erased, so the
// dispatch that would read the buffer it writes (the whole Concat of an
// in-place Concat pair bound to one buffer) is gone.

// Past and present are the same allocation: only the in-place dispatch is
// left.
// CHECK-LABEL: func.func @Shared
// CHECK: vrc.case = "in_place"
// CHECK-NOT: vrc.case = "whole"
// CHECK: return
module attributes {
  vrc.static_device_resource_provenance = {
    discovery_schema_version = 1 : i32,
    schema_version = 1 : i32
  }
} {
  func.func private @udonclResolvePhysicalBuffer(!udonhl.ptr<0>, !udonhl.ptr<0>) -> i32

  func.func @Shared(%past: !udonhl.ptr<0>, %present: !udonhl.ptr<0>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %row = "vrc.assigned_buffer"() <{capacity_bytes = 4 : i64, format = "rgba8_unorm", resource_id = 5 : i32, texture_id = 1 : i32}> : () -> !vrc.buffer
    %past_slot = "udonhl.alloca"(%c4) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %past_status = func.call @udonclResolvePhysicalBuffer(%past, %past_slot) {
      vrc.static_device_operand_provenance = [{
        cases = [{call_path = [], may_be_null = false, resource_ids = [2 : i32], root = @Shared, unknown = false}],
        index = 0 : i32
      }]
    } : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    %past_ok = arith.cmpi eq, %past_status, %c0 : i32
    cf.cond_br %past_ok, ^resolve_present, ^failure
  ^resolve_present:
    %present_slot = "udonhl.alloca"(%c4) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %present_status = func.call @udonclResolvePhysicalBuffer(%present, %present_slot) {
      vrc.static_device_operand_provenance = [{
        cases = [{call_path = [], may_be_null = false, resource_ids = [2 : i32], root = @Shared, unknown = false}],
        index = 0 : i32
      }]
    } : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    %present_ok = arith.cmpi eq, %present_status, %c0 : i32
    cf.cond_br %present_ok, ^compare, ^failure
  ^compare:
    %past_token = "udonhl.load"(%past_slot) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> i32
    %present_token = "udonhl.load"(%present_slot) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> i32
    %past_buffer = builtin.unrealized_conversion_cast %past_token : i32 to !vrc.buffer
    %present_buffer = builtin.unrealized_conversion_cast %present_token : i32 to !vrc.buffer
    %descriptor = vrc.dispatch_desc_create %c1 {kernel_id = 8 : i32, scalar_kinds = ["i32"]} : (i32) -> !vrc.dispatch_desc
    %same = arith.cmpi eq, %past_token, %present_token : i32
    cf.cond_br %same, ^in_place, ^whole
  ^in_place:
    %rows = vrc.dispatch %descriptor(%row, %present_buffer) {
      argument_kinds = ["buffer", "buffer", "i32"],
      global_offset = array<i32: 0, 0, 0>,
      global_size = array<i32: 1, 1, 1>,
      kernel_id = 8 : i32,
      local_size = array<i32: 1, 1, 1>,
      resource_accesses = ["read", "write"],
      vrc.case = "in_place"
    } : (!vrc.dispatch_desc, !vrc.buffer, !vrc.buffer) -> !vrc.event
    cf.br ^done
  ^whole:
    %all = vrc.dispatch %descriptor(%past_buffer, %present_buffer) {
      argument_kinds = ["buffer", "buffer", "i32"],
      global_offset = array<i32: 0, 0, 0>,
      global_size = array<i32: 1, 1, 1>,
      kernel_id = 8 : i32,
      local_size = array<i32: 1, 1, 1>,
      resource_accesses = ["read", "write"],
      vrc.case = "whole"
    } : (!vrc.dispatch_desc, !vrc.buffer, !vrc.buffer) -> !vrc.event
    cf.br ^done
  ^done:
    return
  ^failure:
    return
  }
}

// -----

// Different allocations: only the whole Concat is left.
// CHECK-LABEL: func.func @Separate
// CHECK-NOT: vrc.case = "in_place"
// CHECK: vrc.case = "whole"
module attributes {
  vrc.static_device_resource_provenance = {
    discovery_schema_version = 1 : i32,
    schema_version = 1 : i32
  }
} {
  func.func private @udonclResolvePhysicalBuffer(!udonhl.ptr<0>, !udonhl.ptr<0>) -> i32

  func.func @Separate(%past: !udonhl.ptr<0>, %present: !udonhl.ptr<0>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %row = "vrc.assigned_buffer"() <{capacity_bytes = 4 : i64, format = "rgba8_unorm", resource_id = 5 : i32, texture_id = 1 : i32}> : () -> !vrc.buffer
    %past_slot = "udonhl.alloca"(%c4) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %past_status = func.call @udonclResolvePhysicalBuffer(%past, %past_slot) {
      vrc.static_device_operand_provenance = [{
        cases = [{call_path = [], may_be_null = false, resource_ids = [2 : i32], root = @Separate, unknown = false}],
        index = 0 : i32
      }]
    } : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    %past_ok = arith.cmpi eq, %past_status, %c0 : i32
    cf.cond_br %past_ok, ^resolve_present, ^failure
  ^resolve_present:
    %present_slot = "udonhl.alloca"(%c4) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %present_status = func.call @udonclResolvePhysicalBuffer(%present, %present_slot) {
      vrc.static_device_operand_provenance = [{
        cases = [{call_path = [], may_be_null = false, resource_ids = [3 : i32], root = @Separate, unknown = false}],
        index = 0 : i32
      }]
    } : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    %present_ok = arith.cmpi eq, %present_status, %c0 : i32
    cf.cond_br %present_ok, ^compare, ^failure
  ^compare:
    %past_token = "udonhl.load"(%past_slot) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> i32
    %present_token = "udonhl.load"(%present_slot) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> i32
    %past_buffer = builtin.unrealized_conversion_cast %past_token : i32 to !vrc.buffer
    %present_buffer = builtin.unrealized_conversion_cast %present_token : i32 to !vrc.buffer
    %descriptor = vrc.dispatch_desc_create %c1 {kernel_id = 8 : i32, scalar_kinds = ["i32"]} : (i32) -> !vrc.dispatch_desc
    %same = arith.cmpi eq, %past_token, %present_token : i32
    cf.cond_br %same, ^in_place, ^whole
  ^in_place:
    %rows = vrc.dispatch %descriptor(%row, %present_buffer) {
      argument_kinds = ["buffer", "buffer", "i32"],
      global_offset = array<i32: 0, 0, 0>,
      global_size = array<i32: 1, 1, 1>,
      kernel_id = 8 : i32,
      local_size = array<i32: 1, 1, 1>,
      resource_accesses = ["read", "write"],
      vrc.case = "in_place"
    } : (!vrc.dispatch_desc, !vrc.buffer, !vrc.buffer) -> !vrc.event
    cf.br ^done
  ^whole:
    %all = vrc.dispatch %descriptor(%past_buffer, %present_buffer) {
      argument_kinds = ["buffer", "buffer", "i32"],
      global_offset = array<i32: 0, 0, 0>,
      global_size = array<i32: 1, 1, 1>,
      kernel_id = 8 : i32,
      local_size = array<i32: 1, 1, 1>,
      resource_accesses = ["read", "write"],
      vrc.case = "whole"
    } : (!vrc.dispatch_desc, !vrc.buffer, !vrc.buffer) -> !vrc.event
    cf.br ^done
  ^done:
    return
  ^failure:
    return
  }
}
