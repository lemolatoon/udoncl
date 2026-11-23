// RUN: %vrc-opt --split-input-file --vrc-configure-unity-exports %s | %FileCheck %s
// RUN: %vrc-opt --split-input-file --vrc-configure-unity-exports="finalize-reachability=false" %s | %FileCheck %s --check-prefix=PRELIMINARY

// Every UDONCL_EXPORT is a U# export named after the PascalCase of its
// symbol, with its C parameter names. Helpers become private and
// unreachable functions are erased.
module attributes {udoncl.program = "probe",
    udoncl.unity_library_config = {
      binding = {namespace = "Lemolatoon.Udoncl.Generated", class_name = "ProbeBinding"}}} {
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
// CHECK-SAME: vrc.program = "probe"
// CHECK-SAME: vrc.target = "vrc-pc"
// CHECK-NOT: udoncl.unity_library_config
// CHECK: func.func @read_streamed_text
// CHECK-SAME: vrc.allowed_while_busy
// CHECK-SAME: vrc.export_id =
// CHECK-SAME: vrc.parameter_names = ["offset", "length"]
// CHECK-SAME: vrc.public_name = "ReadStreamedText"
// CHECK: func.func @run_probe
// CHECK-NOT: vrc.allowed_while_busy
// CHECK-SAME: vrc.parameter_names = ["n"]
// CHECK-SAME: vrc.public_name = "RunProbe"
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
module attributes {udoncl.program = "onnx_probe",
    udoncl.unity_library_config = {}} {
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

// A C constructor that is not an export is configured as one, so that the
// Host analyses see it as a root, and marked internal.
module attributes {udoncl.program = "constructor_probe",
    udoncl.unity_library_config = {}} {
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
module attributes {udoncl.program = "destructor_probe",
    udoncl.unity_library_config = {}} {
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

// Without binding names the binding has the default names.
module attributes {udoncl.program = "declared_probe",
    udoncl.unity_library_config = {}} {
  func.func @run() attributes {udoncl.export} {
    return
  }
}

// CHECK-LABEL: module attributes
// CHECK-SAME: vrc.binding_class = "UdonclDeclaredProbeBinding"
// CHECK-SAME: vrc.binding_namespace = "Lemolatoon.Udoncl.Generated"
// CHECK-SAME: vrc.program = "declared_probe"
// CHECK: func.func @run

// -----

// A binding class (--binding-class) keeps the default namespace.
module attributes {udoncl.program = "class_only",
    udoncl.unity_library_config = {binding = {class_name = "CustomBinding"}}} {
  func.func @run() attributes {udoncl.export} {
    return
  }
}

// CHECK-LABEL: module attributes
// CHECK-SAME: vrc.binding_class = "CustomBinding"
// CHECK-SAME: vrc.binding_namespace = "Lemolatoon.Udoncl.Generated"
