module attributes {
  vrc.static_device_resource_discovery = {schema_version = 1 : i32, contexts = [
    {allocation_site = {function = @helper, occurrence = 0 : i32, primitive = "clCreateBuffer"}, call_path = [{caller = @entry, callee = @helper, occurrence = 1 : i32}], context_id = 1 : i32, root = @entry},
    {allocation_site = {function = @direct, occurrence = 0 : i32, primitive = "udonclCreateBufferForQueue"}, call_path = [{caller = @entry, callee = @direct, occurrence = 0 : i32}], context_id = 2 : i32, root = @entry},
    {allocation_site = {function = @helper, occurrence = 0 : i32, primitive = "clCreateBuffer"}, call_path = [{caller = @entry, callee = @helper, occurrence = 0 : i32}], context_id = 0 : i32, root = @entry}
  ]}
} {
}

// -----

module attributes {
  vrc.static_device_resource_discovery = {schema_version = 1 : i32, contexts = []}
} {
}
