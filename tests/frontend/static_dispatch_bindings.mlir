module attributes {
  vrc.static_device_resource_provenance = {
    discovery_schema_version = 1 : i32,
    schema_version = 1 : i32
  }
} {
  func.func private @udonclResolvePhysicalBuffer(!udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
  func.func private @udoncl_mem_physical_token(!udonhl.ptr<0>) -> i32

  func.func private @PhysicalDispatch(%a: !udonhl.ptr<0>, %b: !udonhl.ptr<0>, %c: !udonhl.ptr<0>) {
    %c1 = arith.constant 1 : i32
    %a_token = func.call @udoncl_mem_physical_token(%a) : (!udonhl.ptr<0>) -> i32
    %b_token = func.call @udoncl_mem_physical_token(%b) : (!udonhl.ptr<0>) -> i32
    %c_token = func.call @udoncl_mem_physical_token(%c) : (!udonhl.ptr<0>) -> i32
    %a_buffer = builtin.unrealized_conversion_cast %a_token : i32 to !vrc.buffer
    %b_buffer = builtin.unrealized_conversion_cast %b_token : i32 to !vrc.buffer
    %c_buffer = builtin.unrealized_conversion_cast %c_token : i32 to !vrc.buffer
    %descriptor = vrc.dispatch_desc_create %c1 {kernel_id = 9 : i32, scalar_kinds = ["i32"]} : (i32) -> !vrc.dispatch_desc
    %event = vrc.dispatch %descriptor(%a_buffer, %b_buffer, %c_buffer) {
      argument_kinds = ["buffer", "buffer", "buffer", "i32"],
      global_offset = array<i32: 0, 0, 0>,
      global_size = array<i32: 1, 1, 1>,
      kernel_id = 9 : i32,
      local_size = array<i32: 1, 1, 1>,
      resource_accesses = ["read", "read", "write"]
    } : (!vrc.dispatch_desc, !vrc.buffer, !vrc.buffer, !vrc.buffer) -> !vrc.event
    return
  }

  func.func @RunPhysical(%a: !udonhl.ptr<0>, %b: !udonhl.ptr<0>, %c: !udonhl.ptr<0>) {
    func.call @PhysicalDispatch(%a, %b, %c) {
      vrc.static_device_operand_provenance = [
        {cases = [{call_path = [], may_be_null = false, resource_ids = [1 : i32], root = @RunPhysical, unknown = false}], index = 0 : i32},
        {cases = [{call_path = [], may_be_null = false, resource_ids = [0 : i32], root = @RunPhysical, unknown = false}], index = 1 : i32},
        {cases = [{call_path = [], may_be_null = false, resource_ids = [2 : i32], root = @RunPhysical, unknown = false}], index = 2 : i32}
      ]
    } : (!udonhl.ptr<0>, !udonhl.ptr<0>, !udonhl.ptr<0>) -> ()
    return
  }

  func.func @RunDirect(%dynamic_id: i32) {
    %c1 = arith.constant 1 : i32
    %c5 = arith.constant 5 : i32
    %c16 = arith.constant 16 : i32
    %fixed, %fixed_status = vrc.buffer_alloc %c5, %c16 {format = "rgba8_unorm", usage = "storage"} : (i32, i32) -> (!vrc.buffer, i32)
    %fixed_descriptor = vrc.dispatch_desc_create %c1 {kernel_id = 10 : i32, scalar_kinds = ["i32"]} : (i32) -> !vrc.dispatch_desc
    %fixed_event = vrc.dispatch %fixed_descriptor(%fixed) {
      argument_kinds = ["buffer", "i32"], global_offset = array<i32: 0, 0, 0>,
      global_size = array<i32: 1, 1, 1>, kernel_id = 10 : i32,
      local_size = array<i32: 1, 1, 1>, resource_accesses = ["write"]
    } : (!vrc.dispatch_desc, !vrc.buffer) -> !vrc.event
    %dynamic, %dynamic_status = vrc.buffer_alloc %dynamic_id, %c16 {format = "rgba8_unorm", usage = "storage"} : (i32, i32) -> (!vrc.buffer, i32)
    %dynamic_descriptor = vrc.dispatch_desc_create %c1 {kernel_id = 11 : i32, scalar_kinds = ["i32"]} : (i32) -> !vrc.dispatch_desc
    %dynamic_event = vrc.dispatch %dynamic_descriptor(%dynamic) {
      argument_kinds = ["buffer", "i32"], global_offset = array<i32: 0, 0, 0>,
      global_size = array<i32: 1, 1, 1>, kernel_id = 11 : i32,
      local_size = array<i32: 1, 1, 1>, resource_accesses = ["write"]
    } : (!vrc.dispatch_desc, !vrc.buffer) -> !vrc.event
    return
  }

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

  func.func @RunUnresolved(%resource: !udonhl.ptr<0>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %output = "udonhl.alloca"(%c4) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %status = func.call @udonclResolvePhysicalBuffer(%resource, %output) {
      vrc.static_device_operand_provenance = [{
        cases = [{call_path = [], may_be_null = true, resource_ids = [], root = @RunUnresolved, unknown = false}],
        index = 0 : i32
      }]
    } : (!udonhl.ptr<0>, !udonhl.ptr<0>) -> i32
    %ok = arith.cmpi eq, %status, %c0 : i32
    cf.cond_br %ok, ^success, ^failure
  ^success:
    %token = "udonhl.load"(%output) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> i32
    %buffer = builtin.unrealized_conversion_cast %token : i32 to !vrc.buffer
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
