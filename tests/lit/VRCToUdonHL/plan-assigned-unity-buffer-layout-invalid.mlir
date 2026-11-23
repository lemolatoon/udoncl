// RUN: %vrc-opt --split-input-file --verify-diagnostics --vrc-plan-assigned-unity-buffer-layout %s

module {
  func.func @run() {
    %a = "vrc.assigned_buffer"() <{capacity_bytes = 4 : i64, format = "rgba8_unorm", resource_id = 0 : i32, texture_id = 1 : i32}> : () -> !vrc.buffer
    // expected-error@below {{assigned Unity buffer texture 1 has conflicting resource, capacity or format}}
    %b = "vrc.assigned_buffer"() <{capacity_bytes = 8 : i64, format = "rgba8_unorm", resource_id = 0 : i32, texture_id = 1 : i32}> : () -> !vrc.buffer
    return
  }
}

// -----

module {
  func.func @run() {
    // expected-error@below {{assigned Unity buffer textures must be numbered densely from 1; texture 2 has no predecessor}}
    %a = "vrc.assigned_buffer"() <{capacity_bytes = 4 : i64, format = "rgba8_unorm", resource_id = 0 : i32, texture_id = 2 : i32}> : () -> !vrc.buffer
    return
  }
}

// -----

module {
  func.func @run() {
    %a = "vrc.assigned_buffer"() <{capacity_bytes = 4 : i64, format = "rgba8_unorm", resource_id = 0 : i32, texture_id = 1 : i32}> : () -> !vrc.buffer
    // expected-error@below {{assigned Unity buffers of textures 2 and another share resource 0}}
    %b = "vrc.assigned_buffer"() <{capacity_bytes = 4 : i64, format = "rgba8_unorm", resource_id = 0 : i32, texture_id = 2 : i32}> : () -> !vrc.buffer
    return
  }
}
