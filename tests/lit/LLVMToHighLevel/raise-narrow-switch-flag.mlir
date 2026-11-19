// RUN: %vrc-opt --vrc-raise-llvm-to-scalar %s | %FileCheck %s

// LLVM may switch on a narrow integer (for example a byte). The Udon backend
// compares switch flags as SystemInt32, so raising widens the flag with zero
// extension and widens the case values the same way.

// CHECK-LABEL: func.func @classify(
// CHECK: %[[FLAG:.+]] = arith.extui %{{.+}} : i8 to i32
// CHECK: cf.switch %[[FLAG]] : i32, [
// CHECK-NEXT: default: ^bb{{[0-9]+}},
// CHECK-NEXT: 224: ^bb{{[0-9]+}},
// CHECK-NEXT: 237: ^bb{{[0-9]+}}
// CHECK-NOT: cf.switch %{{.+}} : i8
module {
  llvm.func @classify(%byte: i8) -> i32 {
    llvm.switch %byte : i8, ^default [
      -32: ^lower,
      -19: ^upper
    ]
  ^lower:
    %a = llvm.mlir.constant(1 : i32) : i32
    llvm.return %a : i32
  ^upper:
    %b = llvm.mlir.constant(2 : i32) : i32
    llvm.return %b : i32
  ^default:
    %c = llvm.mlir.constant(0 : i32) : i32
    llvm.return %c : i32
  }
}
