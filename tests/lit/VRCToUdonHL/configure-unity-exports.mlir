// RUN: %vrc-opt --split-input-file --vrc-configure-unity-exports %s | %FileCheck %s
// RUN: %vrc-opt --split-input-file --vrc-configure-unity-exports="finalize-reachability=false" %s | %FileCheck %s --check-prefix=PRELIMINARY

// Schema 2 publishes every UDONCL_EXPORT under the PascalCase of its symbol
// with its C parameter names; an override wins. Helpers become private and
// unreachable functions are erased.
module attributes {udoncl.unity_library_config = {
    schema_version = 2 : i64, program = "probe",
    binding = {namespace = "Lemolatoon.Udoncl.Generated", class_name = "ProbeBinding"},
    export_overrides = [{symbol = "run_probe", public_name = "Run", parameters = ["count"]}]}} {
  func.func @read_streamed_text(%a: i32, %b: i32) -> i32 attributes {
      udoncl.export, udoncl.export_while_busy,
      udoncl.parameter_names = ["offset", "length"]} {
    %sum = func.call @helper(%a, %b) : (i32, i32) -> i32
    return %sum : i32
  }
  func.func @run_probe(%a: i32) attributes {udoncl.export, udoncl.parameter_names = ["n"]} {
    return
  }
  func.func @no_names(%a: i32) attributes {udoncl.export} {
    return
  }
  func.func @helper(%a: i32, %b: i32) -> i32 {
    %sum = arith.addi %a, %b : i32
    return %sum : i32
  }
  func.func @unused() {
    return
  }
}

// CHECK-LABEL: module attributes
// CHECK-SAME: vrc.binding_class = "ProbeBinding"
// CHECK-SAME: vrc.binding_namespace = "Lemolatoon.Udoncl.Generated"
// CHECK-SAME: vrc.target = "vrc-pc"
// CHECK-NOT: udoncl.unity_library_config
// CHECK: func.func @read_streamed_text
// CHECK-SAME: vrc.allowed_while_busy
// CHECK-SAME: vrc.export_id =
// CHECK-SAME: vrc.parameter_names = ["offset", "length"]
// CHECK-SAME: vrc.public_name = "ReadStreamedText"
// CHECK: func.func @run_probe
// CHECK-SAME: vrc.parameter_names = ["count"]
// CHECK-SAME: vrc.public_name = "Run"
// CHECK: func.func @no_names
// CHECK-SAME: vrc.parameter_names = ["arg0"]
// CHECK-SAME: vrc.public_name = "NoNames"
// CHECK: func.func private @helper
// CHECK-NOT: @unused

// Without finalization every function stays in place.
// PRELIMINARY-LABEL: module attributes
// PRELIMINARY-NOT: vrc.target
// PRELIMINARY: func.func @helper
// PRELIMINARY: func.func @unused

// -----

// A generated binding function is reachable from an export but never one.
module attributes {udoncl.unity_library_config = {
    schema_version = 2 : i64, program = "onnx_probe",
    internal_exports = ["ignored"]}} {
  func.func @component_run() -> i32 attributes {udoncl.export} {
    %status = func.call @udon_onnx_binding_enqueue_run() : () -> i32
    return %status : i32
  }
  func.func @udon_onnx_binding_enqueue_run() -> i32 attributes {
      udoncl.export, udoncl.internal_export} {
    %zero = arith.constant 0 : i32
    return %zero : i32
  }
  func.func @udon_onnx_binding_dispose() -> i32 attributes {
      udoncl.export, udoncl.internal_export} {
    %zero = arith.constant 0 : i32
    return %zero : i32
  }
}

// CHECK-LABEL: module attributes
// CHECK: func.func @component_run
// CHECK-SAME: vrc.public_name = "ComponentRun"
// CHECK: func.func private @udon_onnx_binding_enqueue_run
// CHECK-NOT: vrc.export_id
// CHECK-NOT: @udon_onnx_binding_dispose

// -----

// Schema 1 lists every export with its id.
module attributes {udoncl.unity_library_config = {
    schema_version = 1 : i64, program = "listed",
    exports = [{symbol = "listed_run", public_name = "Run", id = 7 : i64, parameters = ["value"], allowed_while_busy = true}]}} {
  func.func @listed_run(%a: f32) {
    return
  }
}

// CHECK-LABEL: module attributes
// CHECK: func.func @listed_run
// CHECK-SAME: vrc.allowed_while_busy
// CHECK-SAME: vrc.export_id = 7 : i32
// CHECK-SAME: vrc.parameter_names = ["value"]
// CHECK-SAME: vrc.public_name = "Run"

// -----

// Static initialization records the constructor and the omitted exports.
module attributes {udoncl.unity_library_config = {
    schema_version = 2 : i64, program = "static_probe",
    static_initialization = {constructor = "probe_initialize", omit_exports = ["probe_dispose"]}}} {
  func.func @probe_initialize() -> i32 attributes {udoncl.export} {
    %zero = arith.constant 0 : i32
    return %zero : i32
  }
  func.func @probe_dispose() attributes {udoncl.export} {
    return
  }
}

// CHECK-LABEL: module attributes
// CHECK-SAME: vrc.static_initialization = {constructor = "probe_initialize", omitted_exports = ["probe_dispose"]}
// CHECK: func.func @probe_initialize
// CHECK-SAME: vrc.public_name = "ProbeInitialize"
// CHECK-SAME: vrc.static_constructor
// CHECK-NOT: @probe_dispose

// -----

// A C constructor that is not an export is configured as one, so that the
// Host analyses see it as a root, and marked internal.
module attributes {udoncl.unity_library_config = {
    schema_version = 2 : i64, program = "constructor_probe"}} {
  func.func private @initialize() -> i32 attributes {udoncl.static_constructor} {
    %zero = arith.constant 0 : i32
    return %zero : i32
  }
  func.func @run() attributes {udoncl.export} {
    return
  }
}

// CHECK-LABEL: module attributes
// CHECK-SAME: vrc.static_initialization = {constructor = "initialize", omitted_exports = []}
// CHECK: func.func @initialize
// CHECK-SAME: vrc.internal_static_constructor
// CHECK-SAME: vrc.public_name = "Initialize"
// CHECK-SAME: vrc.static_constructor
// CHECK: func.func @run

// -----

// An export that is a C destructor is omitted: the artifact is never
// unloaded.
module attributes {udoncl.unity_library_config = {
    schema_version = 2 : i64, program = "destructor_probe"}} {
  func.func @initialize() attributes {udoncl.export, udoncl.static_constructor} {
    return
  }
  func.func @shutdown() attributes {udoncl.export, udoncl.static_destructor} {
    return
  }
}

// CHECK-LABEL: module attributes
// CHECK-SAME: vrc.static_initialization = {constructor = "initialize", omitted_exports = ["shutdown"]}
// CHECK: func.func @initialize
// CHECK-NOT: @shutdown

// -----

// UDONCL_PROGRAM names the program when the configuration does not.
module attributes {udoncl.program = "declared_probe",
    udoncl.unity_library_config = {schema_version = 2 : i64}} {
  func.func @run() attributes {udoncl.export} {
    return
  }
}

// CHECK-LABEL: module attributes
// CHECK-SAME: vrc.program = "declared_probe"
// CHECK: func.func @run
