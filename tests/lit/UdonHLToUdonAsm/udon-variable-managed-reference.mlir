// RUN: %vrc-opt --vrc-lower-udonhl-to-udonasm %s | %FileCheck %s

// A Udon variable holding a managed reference (here a udon_string_t) is a
// heap slot of its Udon type that starts as null; one export stores the
// string, another reads it.

"udonhl.global"() <{sym_name = "message", storage_type = !udonhl.ptr<200>, initial_value = 0 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> {vrc.udon_variable} : () -> ()

// CHECK-DAG: {initial = "null", name = "__vrc_variable_message", type = "SystemString"}
// CHECK-DAG: {initial = "null", name = "__vrc_variable_previous", type = "SystemString"}
// CHECK: udonasm.push "__0_arg0__param"
// CHECK-NEXT: udonasm.push "__vrc_variable_message"
// CHECK-NEXT: udonasm.copy
func.func @keep(%text: !udonhl.ptr<200>) attributes {vrc.export_id = 1 : i32} {
  udonhl.variable_store %text, @message : !udonhl.ptr<200>
  return
}
// CHECK: udonasm.push "__vrc_variable_message"
// CHECK-NEXT: udonasm.push "__vrc_variable_previous"
// CHECK-NEXT: udonasm.copy
"udonhl.global"() <{sym_name = "previous", storage_type = !udonhl.ptr<200>, initial_value = 0 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> {vrc.udon_variable} : () -> ()
func.func @remember() attributes {vrc.export_id = 2 : i32} {
  %text = udonhl.variable_load @message : !udonhl.ptr<200>
  udonhl.variable_store %text, @previous : !udonhl.ptr<200>
  return
}
