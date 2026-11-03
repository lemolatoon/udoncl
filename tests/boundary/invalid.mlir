// RUN: vrc-opt --split-input-file --verify-diagnostics %s
module {
  func.func @bad() {
    // expected-error @below {{callee must resolve to func.func}}
    udonhl.call @missing() {call_policy = #udonhl<call_policy normal>} : () -> ()
    return
  }
}

// -----
module {
  func.func private @callee(i32)
  func.func @bad(%x: f32) {
    // expected-error @below {{callee argument/result types do not match}}
    udonhl.call @callee(%x) {call_policy = #udonhl<call_policy normal>} : (f32) -> ()
    return
  }
}

// -----
module {
  func.func @bad(%d: !vrc.dispatch_desc, %b: !vrc.buffer) {
    // expected-error @below {{ordered resource count must match buffer arguments}}
    %e = vrc.dispatch %d(%b) {kernel_id = 7 : i32, argument_kinds = ["buffer", "buffer"], global_size = array<i32: 1, 1, 1>, local_size = array<i32: 1, 1, 1>, global_offset = array<i32: 0, 0, 0>} : (!vrc.dispatch_desc, !vrc.buffer) -> !vrc.event
    return
  }
}

// -----
module {
  func.func @bad() {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    // expected-error @below {{constant byte size must be positive}}
    %buffer, %status = vrc.buffer_alloc %one, %zero {format = "rgba8_unorm", usage = "storage"} : (i32, i32) -> (!vrc.buffer, i32)
    return
  }
}

// -----
module {
  func.func @bad(%source: !vrc.buffer, %destination: !vrc.buffer) {
    %negative = arith.constant -1 : i32
    %zero = arith.constant 0 : i32
    %bytes = arith.constant 4 : i32
    // expected-error @below {{constant source offset must be non-negative}}
    %event = vrc.buffer_copy %source, %destination, %negative, %zero, %bytes : (!vrc.buffer, !vrc.buffer, i32, i32, i32) -> !vrc.event
    return
  }
}

// -----
module {
  func.func @bad(%d: !vrc.dispatch_desc, %b: !vrc.buffer) {
    // expected-error @below {{NDRange must contain three dimensions}}
    %e = vrc.dispatch %d(%b) {kernel_id = 7 : i32, argument_kinds = ["buffer", "f32"], global_size = array<i32: 1>, local_size = array<i32: 1, 1, 1>, global_offset = array<i32: 0, 0, 0>} : (!vrc.dispatch_desc, !vrc.buffer) -> !vrc.event
    return
  }
}

// -----
module {
  "vrcgpu.kernel"() <{kernel_id = 7 : i32, sym_name = "k", entry_name = "k", source_module = "device", argument_kinds = ["buffer", "f32"], global_size = array<i32: 1, 1, 1>, local_size = array<i32: 1, 1, 1>, global_offset = array<i32: 0, 0, 0>}> ({
    "vrcgpu.resource"() <{sym_name = "arg0", kind = "target", arg_index = 0 : i32, binding = 0 : i32, value_type = f32, format = "rgba8_unorm"}> : () -> ()
    "vrcgpu.resource"() <{sym_name = "arg1", kind = "uniform", arg_index = 1 : i32, binding = 1 : i32, value_type = f32}> : () -> ()
    %v = "vrcgpu.pass"() <{output = @arg0, pass_id = 0 : i32}> ({
      %x = "vrcgpu.uniform_load"() <{resource = @arg1}> : () -> f32
      // expected-error @below {{initial raw_pack encodes one complete f32/i32 word at byte offset zero}}
      %p = "vrcgpu.raw_pack"(%x) <{byte_offset = 1 : i32, format = "rgba8_unorm"}> : (f32) -> vector<4xf32>
      "vrcgpu.output"(%p) : (vector<4xf32>) -> ()
    }) : () -> !vrcgpu.version<0>
  }) : () -> ()
}

// -----
module {
  "vrcgpu.kernel"() <{kernel_id = 7 : i32, sym_name = "k", entry_name = "k", source_module = "device", argument_kinds = ["buffer", "f32"], global_size = array<i32: 1, 1, 1>, local_size = array<i32: 1, 1, 1>, global_offset = array<i32: 0, 0, 0>}> ({
    "vrcgpu.resource"() <{sym_name = "arg0", kind = "target", arg_index = 0 : i32, binding = 0 : i32, value_type = f32, format = "rgba8_unorm"}> : () -> ()
    "vrcgpu.resource"() <{sym_name = "arg1", kind = "uniform", arg_index = 1 : i32, binding = 1 : i32, value_type = f32}> : () -> ()
    // expected-error @below {{operation is not fragment-safe}}
    %v = "vrcgpu.pass"() <{output = @arg0, pass_id = 0 : i32}> ({
      "gpu.barrier"() : () -> ()
      %x = "vrcgpu.uniform_load"() <{resource = @arg1}> : () -> f32
      %p = "vrcgpu.raw_pack"(%x) <{byte_offset = 0 : i32, format = "rgba8_unorm"}> : (f32) -> vector<4xf32>
      "vrcgpu.output"(%p) : (vector<4xf32>) -> ()
    }) : () -> !vrcgpu.version<0>
  }) : () -> ()
}

// -----
module {
  // expected-error @below {{vrcgpu.kernel}}
  "vrcgpu.kernel"() <{kernel_id = 7 : i32, sym_name = "k", entry_name = "k", source_module = "device", argument_kinds = ["buffer", "f32"], global_size = array<i32: 1, 1, 1>, local_size = array<i32: 1, 1, 1>, global_offset = array<i32: 0, 0, 0>}> ({}) : () -> ()
}
