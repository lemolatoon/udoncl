// RUN: %vrc-opt --split-input-file --vrc-resolve-static-kernel-bindings %s | %FileCheck %s

// A dispatch operand the compiler already assigned (hand-written Host MLIR
// names its Device buffers with vrc.assigned_buffer) is the allocation the
// assigned buffer names, also through the casts between the buffer and its
// OpenCL token.

// CHECK-LABEL: func.func @run
// CHECK: = vrc.dispatch %
// CHECK-SAME: vrc.static_kernel_binding_snapshot = {kernel_id = 8 : i32, resources = [{access = "read", kind = "allocation", operand_index = 0 : i32, resource_id = 3 : i32}, {access = "write", kind = "allocation", operand_index = 1 : i32, resource_id = 4 : i32}], scalars = [{descriptor_operand_index = 0 : i32, type = i32}], schema_version = 1 : i32}
module attributes {
  vrc.static_device_resource_provenance = {
    discovery_schema_version = 1 : i32,
    schema_version = 1 : i32
  }
} {
  func.func @run() {
    %c1 = arith.constant 1 : i32
    %input = "vrc.assigned_buffer"() <{capacity_bytes = 4 : i64, format = "rgba8_unorm", resource_id = 3 : i32, texture_id = 1 : i32}> : () -> !vrc.buffer
    %token = builtin.unrealized_conversion_cast %input : !vrc.buffer to i32
    %read = builtin.unrealized_conversion_cast %token : i32 to !vrc.buffer
    %output = "vrc.assigned_buffer"() <{capacity_bytes = 4 : i64, format = "rgba8_unorm", resource_id = 4 : i32, texture_id = 2 : i32}> : () -> !vrc.buffer
    %descriptor = vrc.dispatch_desc_create %c1 {kernel_id = 8 : i32, scalar_kinds = ["i32"]} : (i32) -> !vrc.dispatch_desc
    %event = vrc.dispatch %descriptor(%read, %output) {
      argument_kinds = ["buffer", "buffer", "i32"],
      global_offset = array<i32: 0, 0, 0>,
      global_size = array<i32: 1, 1, 1>,
      kernel_id = 8 : i32,
      local_size = array<i32: 1, 1, 1>,
      resource_accesses = ["read", "write"]
    } : (!vrc.dispatch_desc, !vrc.buffer, !vrc.buffer) -> !vrc.event
    return
  }
}
