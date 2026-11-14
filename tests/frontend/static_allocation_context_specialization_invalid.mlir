// expected-error @below {{planned allocation call site is missing or ambiguous}}
module attributes {
  vrc.static_device_resource_context_plan = {schema_version = 1 : i32, sites = [{
    allocation_site = {function = @leaf, occurrence = 0 : i32, primitive = "clCreateBuffer"},
    contexts = [
      {call_path = [{caller = @Run, callee = @middle, occurrence = 0 : i32}, {caller = @middle, callee = @leaf, occurrence = 0 : i32}], resource_id = 0 : i32, root = @Run},
      {call_path = [{caller = @Run, callee = @middle, occurrence = 0 : i32}, {caller = @middle, callee = @leaf, occurrence = 1 : i32}], resource_id = 1 : i32, root = @Run}
    ],
    disposition = "clone_per_context",
    resource_ids = array<i32: 0, 1>
  }]}
} {
  func.func private @leaf() { return }
  func.func private @middle() { return }
  func.func @Run() attributes {udoncl.export} { return }
}
