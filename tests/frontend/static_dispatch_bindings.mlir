module attributes {
  vrc.static_device_resource_provenance = {
    discovery_schema_version = 1 : i32,
    schema_version = 1 : i32
  }
} {
  func.func private @udonclResolvePhysicalBuffer(!udonhl.ptr<0>, !udonhl.ptr<0>) -> i32

  func.func @Run(%resource: !udonhl.ptr<0>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %output = "udonhl.alloca"(%c4) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %status = func.call @udonclResolvePhysicalBuffer(%resource, %output) {
      vrc.static_device_operand_provenance = [{
        cases = [{call_path = [], may_be_null = true, resource_ids = [2 : i32], root = @Run, unknown = false}],
        index = 0 : i32
      }]
    } : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    %ok = arith.cmpi eq, %status, %c0 : i32
    cf.cond_br %ok, ^success, ^failure
  ^success:
    %token = "udonhl.load"(%output) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> i32
    %dynamic = builtin.unrealized_conversion_cast %token : i32 to !vrc.buffer
    %static = "vrc.static_buffer"() <{byte_size = 16 : i32, resource_id = 9 : i32}> : () -> !vrc.buffer
    %static_token = builtin.unrealized_conversion_cast %static : !vrc.buffer to i32
    %static_roundtrip = builtin.unrealized_conversion_cast %static_token : i32 to !vrc.buffer
    %descriptor = vrc.dispatch_desc_create %c1 {kernel_id = 7 : i32, scalar_kinds = ["i32"]} : (i32) -> !vrc.dispatch_desc
    %event = vrc.dispatch %descriptor(%static_roundtrip, %dynamic) {
      argument_kinds = ["buffer", "buffer", "i32"],
      global_offset = array<i32: 0, 0, 0>,
      global_size = array<i32: 1, 1, 1>,
      kernel_id = 7 : i32,
      local_size = array<i32: 1, 1, 1>,
      resource_accesses = ["read", "write"]
    } : (!vrc.dispatch_desc, !vrc.buffer, !vrc.buffer) -> !vrc.event
    return
  ^failure:
    return
  }
}
