// RUN: %vrc-opt --vrc-lower-udonhl-to-udonasm %s | %FileCheck %s

// A global marked vrc.udon_variable is a named heap slot that starts with the
// global's initializer. It takes no raw heap cell.

"udonhl.global"() <{sym_name = "used", storage_type = i32, initial_value = 5 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> {vrc.udon_variable} : () -> ()

// CHECK-DAG: {initial = "5", name = "__vrc_variable_used", type = "SystemInt32"}
// CHECK-NOT: __vrc_raw_heap

// A loaded value is read from the variable itself, and a sum that is stored
// back is computed into it: the load and the store emit nothing, and later
// uses of the sum read the variable.
// CHECK: udonasm.push "__vrc_variable_used"
// CHECK-NEXT: udonasm.push "__0_arg0__param"
// CHECK-NEXT: udonasm.push "__vrc_variable_used"
// CHECK-NEXT: udonasm.extern "SystemInt32.__op_Addition__SystemInt32_SystemInt32__SystemInt32"
// CHECK-NEXT: udonasm.push "__vrc_variable_used"
// CHECK-NEXT: udonasm.push "__0_arg1__param"
// CHECK-NEXT: udonasm.push "[[FULL:[^"]+]]"
// CHECK-NEXT: udonasm.extern "SystemInt32.__op_GreaterThanOrEqual__SystemInt32_SystemInt32__SystemBoolean"
func.func @charge(%cost: i32, %budget: i32) -> i1 attributes {vrc.export_id = 1 : i32} {
  %before = udonhl.variable_load @used : i32
  %after = arith.addi %before, %cost : i32
  udonhl.variable_store %after, @used : i32
  %full = arith.cmpi sge, %after, %budget : i32
  return %full : i1
}

// A value that is still used after the variable is overwritten is copied
// out, and a value defined elsewhere is copied in.
// CHECK: udonasm.push "__vrc_variable_used"
// CHECK-NEXT: udonasm.push "[[OLD:[^"]+]]"
// CHECK-NEXT: udonasm.copy
// CHECK-NEXT: udonasm.push "__1_arg0__param"
// CHECK-NEXT: udonasm.push "__vrc_variable_used"
// CHECK-NEXT: udonasm.copy
// CHECK-NEXT: udonasm.push "[[OLD]]"
// CHECK-NEXT: udonasm.push "__0___0_replace__ret"
// CHECK-NEXT: udonasm.copy
func.func @replace(%value: i32) -> i32 attributes {vrc.export_id = 2 : i32} {
  %old = udonhl.variable_load @used : i32
  udonhl.variable_store %value, @used : i32
  return %old : i32
}

func.func private @reset() {
  %zero = arith.constant 0 : i32
  udonhl.variable_store %zero, @used : i32
  return
}

// A call may write the variable, so a value used after it is copied out
// first.
// CHECK: udonasm.push "__vrc_variable_used"
// CHECK-NEXT: udonasm.push "[[KEPT:[^"]+]]"
// CHECK-NEXT: udonasm.copy
// CHECK: udonasm.jump {{[0-9]+}}
// CHECK-NEXT: udonasm.push "[[KEPT]]"
// CHECK-NEXT: udonasm.push "__2_arg0__param"
// CHECK-NEXT: udonasm.push "{{[^"]+}}"
// CHECK-NEXT: udonasm.extern "SystemInt32.__op_Addition__SystemInt32_SystemInt32__SystemInt32"
func.func @across_call(%value: i32) -> i32 attributes {vrc.export_id = 3 : i32} {
  %old = udonhl.variable_load @used : i32
  call @reset() : () -> ()
  %sum = arith.addi %old, %value : i32
  return %sum : i32
}

// The variable is read between the definition of the stored value and the
// store, so the value cannot be computed into the variable.
// CHECK: udonasm.push "__3_arg0__param"
// CHECK-NEXT: udonasm.push "{{[^"]+}}"
// CHECK-NEXT: udonasm.push "[[NEXT:[^"]+]]"
// CHECK-NEXT: udonasm.extern "SystemInt32.__op_Addition__SystemInt32_SystemInt32__SystemInt32"
// CHECK-NEXT: udonasm.push "__vrc_variable_used"
// CHECK-NEXT: udonasm.push "[[READ:[^"]+]]"
// CHECK-NEXT: udonasm.copy
// CHECK-NEXT: udonasm.push "[[NEXT]]"
// CHECK-NEXT: udonasm.push "__vrc_variable_used"
// CHECK-NEXT: udonasm.copy
// CHECK-NEXT: udonasm.push "[[READ]]"
// CHECK-NEXT: udonasm.push "[[NEXT]]"
func.func @read_before_store(%value: i32) -> i32 attributes {vrc.export_id = 4 : i32} {
  %one = arith.constant 1 : i32
  %next = arith.addi %value, %one : i32
  %old = udonhl.variable_load @used : i32
  udonhl.variable_store %next, @used : i32
  %sum = arith.addi %old, %next : i32
  return %sum : i32
}
