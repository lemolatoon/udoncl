// RUN: %vrc-opt --vrc-lower-udonhl-to-udonasm %s | %FileCheck %s

// A switch with five or more cases is a balanced decision tree: a
// less-than comparison with the middle case splits the sorted cases until
// one equality comparison remains. Every comparison reads its case value
// from a literal slot; none is assembled digit by digit at run time.

// CHECK-DAG: {initial = "300", name = "[[MIDDLE:__vrc_constant_[0-9]+]]", type = "SystemInt32"}
// CHECK-DAG: {initial = "200", name = "[[LOWER:__vrc_constant_[0-9]+]]", type = "SystemInt32"}
// CHECK-DAG: {initial = "100", name = "[[FIRST:__vrc_constant_[0-9]+]]", type = "SystemInt32"}
// CHECK-NOT: name = "__vrc_switch_case_constant"

// CHECK: udonasm.push "__0_arg0__param"
// CHECK-NEXT: udonasm.push "[[MIDDLE]]"
// CHECK-NEXT: udonasm.push "__vrc_switch_condition"
// CHECK-NEXT: udonasm.extern "SystemInt32.__op_LessThan__SystemInt32_SystemInt32__SystemBoolean"
// CHECK-NEXT: udonasm.push "__vrc_switch_condition"
// CHECK-NEXT: udonasm.jump_if_false {{[0-9]+}}
// CHECK-NEXT: udonasm.push "__0_arg0__param"
// CHECK-NEXT: udonasm.push "[[LOWER]]"
// CHECK-NEXT: udonasm.push "__vrc_switch_condition"
// CHECK-NEXT: udonasm.extern "SystemInt32.__op_LessThan__SystemInt32_SystemInt32__SystemBoolean"
// CHECK-NEXT: udonasm.push "__vrc_switch_condition"
// CHECK-NEXT: udonasm.jump_if_false {{[0-9]+}}
// CHECK-NEXT: udonasm.push "__0_arg0__param"
// CHECK-NEXT: udonasm.push "[[FIRST]]"
// CHECK-NEXT: udonasm.push "__vrc_switch_condition"
// CHECK-NEXT: udonasm.extern "SystemInt32.__op_Equality__SystemInt32_SystemInt32__SystemBoolean"
func.func @select(%flag: i32) -> i32 attributes {vrc.export_id = 1 : i32} {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  cf.switch %flag : i32, [
    default: ^other,
    100: ^hit,
    200: ^hit,
    300: ^hit,
    400: ^hit,
    500: ^hit
  ]
^hit:
  return %one : i32
^other:
  return %zero : i32
}
