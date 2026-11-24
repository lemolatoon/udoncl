// RUN: udoncl-opt --vrc-classify-onnx-public-device-resources --split-input-file --verify-diagnostics %s

module attributes {
  udoncl.onnx_binding = {
    inputs = [{capacity_bytes = 16 : i64, data_type = "float32", index = 0 : i32, logical_bytes = 16 : i64, name = "input"}],
    outputs = []
  },
  vrc.static_device_resource_discovery = {contexts = [{context_id = 4 : i32}]}
} {
  func.func private @udononnxLegacySessionBindInput(i32, i32, i32) -> i32
  func.func private @consume(i32)
  func.func @DynamicIndex(%session: i32, %index: i32, %tensor: i32) {
    // expected-error@+1 {{ONNX bind index must be constant}}
    %status = func.call @udononnxLegacySessionBindInput(%session, %index, %tensor) {vrc.static_device_call_site = {callee = @udononnxLegacySessionBindInput, occurrence = 0 : i32}} : (i32, i32, i32) -> i32
    func.call @consume(%tensor) {vrc.static_device_operand_provenance = [{cases = [{call_path = [{caller = @DynamicIndex, callee = @udononnxLegacySessionBindInput, occurrence = 0 : i32}], may_be_null = false, resource_ids = [4 : i32], root = @DynamicIndex, unknown = false}], index = 0 : i32}]} : (i32) -> ()
    return
  }
}

// -----

module attributes {
  udoncl.onnx_binding = {
    inputs = [{capacity_bytes = 16 : i64, data_type = "float32", index = 0 : i32, logical_bytes = 16 : i64, name = "input"}],
    outputs = []
  },
  vrc.static_device_resource_discovery = {contexts = [{context_id = 4 : i32}, {context_id = 5 : i32}]}
} {
  func.func private @udononnxLegacySessionBindInput(i32, i32, i32) -> i32
  func.func private @consume(i32)
  func.func @Ambiguous(%session: i32, %tensor: i32) {
    %index = arith.constant 0 : i32
    %status = func.call @udononnxLegacySessionBindInput(%session, %index, %tensor) {vrc.static_device_call_site = {callee = @udononnxLegacySessionBindInput, occurrence = 2 : i32}} : (i32, i32, i32) -> i32
    func.call @consume(%tensor) {vrc.static_device_operand_provenance = [{cases = [{call_path = [{caller = @Ambiguous, callee = @udononnxLegacySessionBindInput, occurrence = 2 : i32}], may_be_null = false, resource_ids = [4 : i32], root = @Ambiguous, unknown = false}], index = 0 : i32}]} : (i32) -> ()
    // expected-error@+1 {{ONNX bind site has ambiguous resource identity}}
    func.call @consume(%tensor) {vrc.static_device_operand_provenance = [{cases = [{call_path = [{caller = @Ambiguous, callee = @udononnxLegacySessionBindInput, occurrence = 2 : i32}], may_be_null = false, resource_ids = [5 : i32], root = @Ambiguous, unknown = false}], index = 0 : i32}]} : (i32) -> ()
    return
  }
}
