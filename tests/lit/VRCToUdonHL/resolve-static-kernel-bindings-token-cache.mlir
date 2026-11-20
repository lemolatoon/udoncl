// RUN: %vrc-opt --split-input-file --verify-diagnostics --vrc-resolve-static-kernel-bindings %s | %FileCheck %s

// A run keeps a resolved physical token in a global marked
// `vrc.resolved_token_cache` and, while its key is unchanged, reads the global
// instead of resolving again. The dispatch reading the cache has the
// identity of the resolve whose result is the cache's only store; that
// resolve must succeed before the store, not before the dispatch.

// CHECK-LABEL: func.func @Run
// CHECK: = vrc.dispatch %
// CHECK-SAME: kind = "allocation"
// CHECK-SAME: operand_index = 0 : i32, resource_id = 2 : i32
module attributes {
  vrc.static_device_resource_provenance = {
    discovery_schema_version = 1 : i32,
    schema_version = 1 : i32
  }
} {
  "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, initial_value = 0 : i32, linkage = "internal", storage_type = i32, sym_name = "token_cache"}> {vrc.resolved_token_cache} : () -> ()
  "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, initial_value = 0 : i32, linkage = "internal", storage_type = i32, sym_name = "token_key"}> : () -> ()
  func.func private @udonclResolvePhysicalBuffer(!udonhl.ptr<0>, !udonhl.ptr<0>) -> i32

  func.func @Run(%resource: !udonhl.ptr<0>, %key: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %key_address = udonhl.address_of @token_key : !udonhl.ptr<0>
    %cached_key = "udonhl.load"(%key_address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> i32
    %hit = arith.cmpi eq, %cached_key, %key : i32
    cf.cond_br %hit, ^ready, ^resolve
  ^resolve:
    %output = "udonhl.alloca"(%c4) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %status = func.call @udonclResolvePhysicalBuffer(%resource, %output) {
      vrc.static_device_operand_provenance = [{
        cases = [{call_path = [], may_be_null = false, resource_ids = [2 : i32], root = @Run, unknown = false}],
        index = 0 : i32
      }]
    } : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    %ok = arith.cmpi eq, %status, %c0 : i32
    cf.cond_br %ok, ^store, ^failure
  ^store:
    %token = "udonhl.load"(%output) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> i32
    %cache_address = udonhl.address_of @token_cache : !udonhl.ptr<0>
    "udonhl.store"(%token, %cache_address) <{alignment = 4 : i64}> : (i32, !udonhl.ptr<0>) -> ()
    "udonhl.store"(%key, %key_address) <{alignment = 4 : i64}> : (i32, !udonhl.ptr<0>) -> ()
    cf.br ^ready
  ^ready:
    %read_address = udonhl.address_of @token_cache : !udonhl.ptr<0>
    %cached = "udonhl.load"(%read_address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> i32
    %buffer = builtin.unrealized_conversion_cast %cached : i32 to !vrc.buffer
    %descriptor = vrc.dispatch_desc_create %c1 {kernel_id = 8 : i32, scalar_kinds = ["i32"]} : (i32) -> !vrc.dispatch_desc
    %event = vrc.dispatch %descriptor(%buffer) {
      argument_kinds = ["buffer", "i32"],
      global_offset = array<i32: 0, 0, 0>,
      global_size = array<i32: 1, 1, 1>,
      kernel_id = 8 : i32,
      local_size = array<i32: 1, 1, 1>,
      resource_accesses = ["write"]
    } : (!vrc.dispatch_desc, !vrc.buffer) -> !vrc.event
    return
  ^failure:
    return
  }
}

// -----

// A global without the marker is not a cache: its load has no resolve writer.

module attributes {
  vrc.static_device_resource_provenance = {
    discovery_schema_version = 1 : i32,
    schema_version = 1 : i32
  }
} {
  "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, initial_value = 0 : i32, linkage = "internal", storage_type = i32, sym_name = "plain"}> : () -> ()

  func.func @Plain() {
    %c1 = arith.constant 1 : i32
    %address = udonhl.address_of @plain : !udonhl.ptr<0>
    %token = "udonhl.load"(%address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> i32
    %buffer = builtin.unrealized_conversion_cast %token : i32 to !vrc.buffer
    %descriptor = vrc.dispatch_desc_create %c1 {kernel_id = 8 : i32, scalar_kinds = ["i32"]} : (i32) -> !vrc.dispatch_desc
    // expected-error @+1 {{static dispatch physical output cell must have one resolve writer}}
    %event = vrc.dispatch %descriptor(%buffer) {
      argument_kinds = ["buffer", "i32"],
      global_offset = array<i32: 0, 0, 0>,
      global_size = array<i32: 1, 1, 1>,
      kernel_id = 8 : i32,
      local_size = array<i32: 1, 1, 1>,
      resource_accesses = ["write"]
    } : (!vrc.dispatch_desc, !vrc.buffer) -> !vrc.event
    return
  }
}
