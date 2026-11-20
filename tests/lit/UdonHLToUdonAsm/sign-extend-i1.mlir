// RUN: %vrc-opt --vrc-lower-udonhl-to-udonasm %s | %FileCheck %s

// The only bit of an i1 is its sign bit, so sign extension selects -1 or 0.

// CHECK-DAG: {initial = "-1", name = "[[ONES:__vrc_constant_[0-9]+]]", type = "SystemInt32"}
// CHECK-DAG: {initial = "0", name = "[[ZERO:__vrc_constant_[0-9]+]]", type = "SystemInt32"}

// CHECK: udonasm.extern "SystemInt32.__op_LessThan__SystemInt32_SystemInt32__SystemBoolean"
// CHECK-NEXT: udonasm.push "[[FLAG:[^"]+]]"
// CHECK-NEXT: udonasm.jump_if_false [[FALSE:[0-9]+]]
// CHECK-NEXT: udonasm.push "[[ONES]]"
// CHECK-NEXT: udonasm.push "[[MASK:[^"]+]]"
// CHECK-NEXT: udonasm.copy
// CHECK-NEXT: udonasm.jump {{[0-9]+}}
// CHECK-NEXT: udonasm.push "[[ZERO]]"
// CHECK-NEXT: udonasm.push "[[MASK]]"
// CHECK-NEXT: udonasm.copy
func.func @mask(%a: i32, %b: i32) -> i32 attributes {vrc.export_id = 1 : i32} {
  %less = arith.cmpi slt, %a, %b : i32
  %mask = arith.extsi %less : i1 to i32
  return %mask : i32
}
