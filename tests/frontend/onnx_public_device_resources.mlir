module attributes {
  udoncl.onnx_binding = {
    inputs = [
      {capacity_bytes = 16 : i64, data_type = "float32", index = 0 : i32, logical_bytes = 16 : i64, name = "unused_input"},
      {capacity_bytes = 48 : i64, data_type = "int64", index = 1 : i32, logical_bytes = 24 : i64, name = "tokens"}
    ],
    outputs = [
      {capacity_bytes = 48 : i64, data_type = "float32", index = 0 : i32, logical_bytes = 20 : i64, name = "scores"}
    ]
  },
  vrc.static_device_resource_discovery = {
    contexts = [
      {context_id = 11 : i32},
      {context_id = 7 : i32},
      {context_id = 9 : i32}
    ]
  }
} {
  func.func private @udononnxSessionBindInput(i32, i32, i32) -> i32
  func.func private @udononnxSessionBindOutput(i32, i32, i32) -> i32
  func.func private @consume(i32)

  func.func @Configure(%session: i32, %input: i32, %output: i32, %private: i32) {
    %input_index = arith.constant 1 : i32
    %input_status = func.call @udononnxSessionBindInput(%session, %input_index, %input) {
      vrc.static_device_call_site = {callee = @udononnxSessionBindInput, occurrence = 3 : i32}
    } : (i32, i32, i32) -> i32
    %output_index = arith.constant 0 : i32
    %output_status = func.call @udononnxSessionBindOutput(%session, %output_index, %output) {
      vrc.static_device_call_site = {callee = @udononnxSessionBindOutput, occurrence = 8 : i32}
    } : (i32, i32, i32) -> i32
    func.call @consume(%input) {
      vrc.static_device_operand_provenance = [{
        cases = [{call_path = [{caller = @Configure, callee = @udononnxSessionBindInput, occurrence = 3 : i32}], may_be_null = false, resource_ids = [7 : i32], root = @Configure, unknown = false}],
        index = 0 : i32
      }]
    } : (i32) -> ()
    func.call @consume(%output) {
      vrc.static_device_operand_provenance = [{
        cases = [{call_path = [{caller = @Configure, callee = @udononnxSessionBindOutput, occurrence = 8 : i32}], may_be_null = false, resource_ids = [9 : i32], root = @Configure, unknown = false}],
        index = 0 : i32
      }]
    } : (i32) -> ()
    // Resource 11 deliberately has the same capacity as the public resources,
    // but no ONNX bind provenance and must remain unclassified.
    func.call @consume(%private) : (i32) -> ()
    return
  }
}
