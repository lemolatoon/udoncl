// expected-error @below {{static device resource provenance requires discovery schema version 1}}
module {
  func.func @Run() attributes {udoncl.export} {
    return
  }
}

// -----

module attributes {vrc.static_device_resource_discovery = {contexts = [], schema_version = 1 : i32}} {
  func.func private @clCreateBuffer() -> i32
  func.func @Run() attributes {udoncl.export} {
    // expected-error @below {{allocation is missing static device allocation-site metadata}}
    %buffer = func.call @clCreateBuffer() : () -> i32
    return
  }
}
