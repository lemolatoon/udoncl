// RUN: vrc-opt --vrc-raise-llvm-to-scalar --split-input-file --verify-diagnostics %s
module {
  // expected-error @below {{cannot raise llvm.func: pointer/aggregate ABI requires memory/provenance recovery}}
  llvm.func @pointer(%p: !llvm.ptr) {
    llvm.return
  }
}

// -----
module {
  llvm.func @exact_division(%a: i32, %b: i32) -> i32 {
    // expected-error @below {{cannot raise llvm.udiv: poison-generating flag has no implemented mapping}}
    %r = llvm.udiv exact %a, %b : i32
    llvm.return %r : i32
  }
}

// -----
module {
  llvm.func @unpromoted(%a: i32) -> i32 {
    %one = llvm.mlir.constant(1 : i32) : i32
    // expected-error @below {{cannot raise llvm.alloca: non-scalar result requires memory/provenance recovery}}
    %slot = llvm.alloca %one x i32 : (i32) -> !llvm.ptr
    llvm.store %a, %slot : i32, !llvm.ptr
    llvm.return %a : i32
  }
}

// -----
module {
  // expected-error @below {{cannot raise llvm.func: only C calling convention is proven}}
  llvm.func spir_kernelcc @kernel() {
    llvm.return
  }
}
