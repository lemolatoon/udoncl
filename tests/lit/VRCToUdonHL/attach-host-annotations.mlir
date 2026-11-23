// RUN: %vrc-opt --split-input-file --vrc-attach-host-annotations %s | %FileCheck %s

// The annotations of the linked Host LLVM module mark the recovered symbols,
// and the module attribute that carried them is consumed.

// CHECK-LABEL: module {
// CHECK-NOT: udoncl.host_annotations
// CHECK: func.func @read_text(%{{.*}}: i32, %{{.*}}: i32) -> i32 attributes {udoncl.export, udoncl.export_while_busy, udoncl.parameter_names = ["offset", "length"]}
// CHECK: func.func @run() attributes {udoncl.export}
// CHECK: func.func @setup() attributes {udoncl.static_constructor}
// CHECK: func.func @teardown() attributes {udoncl.static_destructor}
// A function recovery reshaped keeps its default parameter names.
// CHECK: func.func @reshaped(%{{.*}}: i32) attributes {udoncl.export}
module attributes {udoncl.host_annotations = {
    constructors = ["setup"], destructors = ["teardown"],
    exports = ["read_text", "run", "reshaped"],
    parameter_names = {read_text = ["offset", "length"], reshaped = ["a", "b"]},
    while_busy = ["read_text"]}} {
  func.func @read_text(%a: i32, %b: i32) -> i32 {
    return %a : i32
  }
  func.func @run() {
    return
  }
  func.func @setup() {
    return
  }
  func.func @teardown() {
    return
  }
  func.func @reshaped(%a: i32) {
    return
  }
}
