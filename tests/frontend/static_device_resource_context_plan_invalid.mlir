// expected-error @below {{static Device context IDs must be dense from zero}}
module attributes {
  vrc.static_device_resource_discovery = {schema_version = 1 : i32, contexts = [
    {allocation_site = {function = @entry, occurrence = 0 : i32, primitive = "clCreateBuffer"}, call_path = [], context_id = 1 : i32, root = @entry}
  ]}
} {
}

// -----

// expected-error @below {{static Device allocation context has an invalid direct call path}}
module attributes {
  vrc.static_device_resource_discovery = {schema_version = 1 : i32, contexts = [
    {allocation_site = {function = @leaf, occurrence = 0 : i32, primitive = "clCreateBuffer"}, call_path = [{caller = @wrong, callee = @leaf, occurrence = 0 : i32}], context_id = 0 : i32, root = @entry}
  ]}
} {
}
