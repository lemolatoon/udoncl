// RUN: %vrc-opt --vrc-thread-constant-branches %s | %FileCheck %s

// `flag = a && b; if (flag) ...` recovers to a decision block that compares
// an integer block argument with zero. The edges that pass the constants 0
// and 1 know the successor and skip the block; the forwarded argument %carry
// is replaced by what each edge passes. With every edge decided the block is
// erased.

// CHECK-LABEL: func.func @short_circuit
// CHECK-SAME: (%[[A:.*]]: i1, %[[B:.*]]: i1, %[[X:.*]]: i32, %[[Y:.*]]: i32)
// CHECK:   cf.cond_br %[[A]], ^[[RHS:bb[0-9]+]], ^[[ELSE:bb[0-9]+]](%[[X]] : i32)
// CHECK: ^[[RHS]]:
// CHECK:   cf.cond_br %[[B]], ^[[THEN:bb[0-9]+]](%[[Y]] : i32), ^[[ELSE]](%[[Y]] : i32)
// CHECK-NOT: arith.cmpi
// CHECK: ^[[THEN]](%{{.*}}: i32):
// CHECK: ^[[ELSE]](%{{.*}}: i32):
func.func @short_circuit(%a: i1, %b: i1, %x: i32, %y: i32) -> i32 {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  cf.cond_br %a, ^rhs, ^decision(%zero, %x : i32, i32)
^rhs:
  cf.cond_br %b, ^decision(%one, %y : i32, i32), ^decision(%zero, %y : i32, i32)
^decision(%flag: i32, %carry: i32):
  %taken = arith.cmpi ne, %flag, %zero : i32
  cf.cond_br %taken, ^then(%carry : i32), ^else(%carry : i32)
^then(%t: i32):
  %sum = arith.addi %t, %one : i32
  return %sum : i32
^else(%e: i32):
  return %e : i32
}

// An i1 block argument that is the condition itself is decided the same way.
// The edge from ^dynamic passes a run-time value and keeps the block.

// CHECK-LABEL: func.func @boolean_argument
// CHECK-SAME: (%[[A:.*]]: i1, %[[B:.*]]: i1)
// CHECK:   cf.cond_br %[[A]], ^[[DYNAMIC:bb[0-9]+]], ^[[ELSE:bb[0-9]+]]
// CHECK: ^[[DYNAMIC]]:
// CHECK:   cf.br ^[[DECISION:bb[0-9]+]](%[[B]] : i1)
// CHECK: ^[[DECISION]](%[[FLAG:.*]]: i1):
// CHECK:   cf.cond_br %[[FLAG]], ^[[THEN:bb[0-9]+]], ^[[ELSE]]
func.func @boolean_argument(%a: i1, %b: i1) -> i32 {
  %false = arith.constant false
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  cf.cond_br %a, ^dynamic, ^decision(%false : i1)
^dynamic:
  cf.br ^decision(%b : i1)
^decision(%flag: i1):
  cf.cond_br %flag, ^then, ^else
^then:
  return %one : i32
^else:
  return %zero : i32
}

// A decision block whose argument is read elsewhere, or that does other
// work, is not skipped.

// CHECK-LABEL: func.func @argument_escapes
// CHECK:   cf.br ^[[DECISION:bb[0-9]+]](%{{.*}} : i32)
// CHECK: ^[[DECISION]](%[[FLAG:.*]]: i32):
// CHECK:   return %[[FLAG]] : i32
func.func @argument_escapes() -> i32 {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  cf.br ^decision(%one : i32)
^decision(%flag: i32):
  %taken = arith.cmpi ne, %flag, %zero : i32
  cf.cond_br %taken, ^then, ^else
^then:
  return %flag : i32
^else:
  return %zero : i32
}

// CHECK-LABEL: func.func @other_work
// CHECK:   cf.br ^[[DECISION:bb[0-9]+]](%{{.*}} : i32)
// CHECK: ^[[DECISION]](%{{.*}}: i32):
// CHECK:   call @effect()
func.func private @effect()
func.func @other_work() -> i32 {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  cf.br ^decision(%one : i32)
^decision(%flag: i32):
  call @effect() : () -> ()
  %taken = arith.cmpi ne, %flag, %zero : i32
  cf.cond_br %taken, ^then, ^else
^then:
  return %one : i32
^else:
  return %zero : i32
}
