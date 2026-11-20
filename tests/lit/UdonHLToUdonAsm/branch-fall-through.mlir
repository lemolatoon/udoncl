// RUN: %vrc-opt --vrc-lower-udonhl-to-udonasm %s | %FileCheck %s

// Blocks are emitted in function order, so an edge to the next block needs
// no jump: after its block-argument copies control falls through. A false
// edge without copies needs no trampoline either; the conditional jump
// targets the successor itself.

// ^entry: the false edge carries copies and falls through into ^loop.
// CHECK: udonasm.extern "SystemInt32.__op_LessThan__SystemInt32_SystemInt32__SystemBoolean"
// CHECK-NEXT: udonasm.push "{{[^"]+}}"
// CHECK-NEXT: udonasm.jump_if_false {{[0-9]+}}
// CHECK-NEXT: udonasm.push "{{[^"]+}}"
// CHECK-NEXT: udonasm.push "{{[^"]+}}"
// CHECK-NEXT: udonasm.copy
// CHECK-NEXT: udonasm.jump [[EXIT:[0-9]+]]
// CHECK-NEXT: udonasm.push "{{[^"]+}}"
// CHECK-NEXT: udonasm.push "{{[^"]+}}"
// CHECK-NEXT: udonasm.copy
// CHECK-NEXT: udonasm.push "{{[^"]+}}"
// CHECK-NEXT: udonasm.push "{{[^"]+}}"
// CHECK-NEXT: udonasm.copy

// ^loop: the false edge to ^exit carries a copy, so the true edge keeps its
// jump over it.
// CHECK-NEXT: udonasm.push "{{[^"]+}}"
// CHECK-NEXT: udonasm.push "{{[^"]+}}"
// CHECK-NEXT: udonasm.push "{{[^"]+}}"
// CHECK-NEXT: udonasm.extern "SystemInt32.__op_LessThan__SystemInt32_SystemInt32__SystemBoolean"
// CHECK-NEXT: udonasm.push "{{[^"]+}}"
// CHECK-NEXT: udonasm.jump_if_false {{[0-9]+}}
// CHECK-NEXT: udonasm.jump {{[0-9]+}}
// CHECK-NEXT: udonasm.push "{{[^"]+}}"
// CHECK-NEXT: udonasm.push "{{[^"]+}}"
// CHECK-NEXT: udonasm.copy
// CHECK-NEXT: udonasm.jump [[EXIT]]

// ^body: neither edge carries copies. The conditional jump targets ^skip
// directly and the true edge falls through into ^add.
// CHECK-NEXT: udonasm.push "{{[^"]+}}"
// CHECK-NEXT: udonasm.push "{{[^"]+}}"
// CHECK-NEXT: udonasm.push "{{[^"]+}}"
// CHECK-NEXT: udonasm.extern "SystemInt32.__op_LogicalAnd__SystemInt32_SystemInt32__SystemInt32"
// CHECK-NEXT: udonasm.push "{{[^"]+}}"
// CHECK-NEXT: udonasm.push "{{[^"]+}}"
// CHECK-NEXT: udonasm.push "{{[^"]+}}"
// CHECK-NEXT: udonasm.extern "SystemInt32.__op_Inequality__SystemInt32_SystemInt32__SystemBoolean"
// CHECK-NEXT: udonasm.push "{{[^"]+}}"
// CHECK-NEXT: udonasm.jump_if_false [[SKIP:[0-9]+]]
// CHECK-NEXT: udonasm.push "{{[^"]+}}"
// CHECK-NEXT: udonasm.push "{{[^"]+}}"
// CHECK-NEXT: udonasm.push "{{[^"]+}}"
// CHECK-NEXT: udonasm.extern "SystemInt32.__op_Addition__SystemInt32_SystemInt32__SystemInt32"
func.func @sum_odd(%n: i32) -> i32 attributes {vrc.export_id = 1 : i32} {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %negative = arith.cmpi slt, %n, %zero : i32
  cf.cond_br %negative, ^exit(%zero : i32), ^loop(%zero, %zero : i32, i32)
^loop(%index: i32, %sum: i32):
  %more = arith.cmpi slt, %index, %n : i32
  cf.cond_br %more, ^body, ^exit(%sum : i32)
^body:
  %low = arith.andi %index, %one : i32
  %odd = arith.cmpi ne, %low, %zero : i32
  cf.cond_br %odd, ^add, ^skip
^add:
  %added = arith.addi %sum, %index : i32
  cf.br ^latch(%added : i32)
^skip:
  cf.br ^latch(%sum : i32)
^latch(%carried: i32):
  %next = arith.addi %index, %one : i32
  cf.br ^loop(%next, %carried : i32, i32)
^exit(%result: i32):
  return %result : i32
}
