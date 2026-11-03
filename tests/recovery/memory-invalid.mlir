// RUN: vrc-opt --vrc-raise-llvm-memory --split-input-file --verify-diagnostics %s
module {
  llvm.func @dynamic(%count: i32) {
    // expected-error @below {{allocation count must be a positive static constant}}
    %a = llvm.alloca %count x i32 : (i32) -> !llvm.ptr
    llvm.return
  }
}

// -----
module {
  llvm.func @uninitialized() -> i32 {
    %one = llvm.mlir.constant(1 : i32) : i32
    %a = llvm.alloca %one x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    // expected-error @below {{definite initialization is not proven}}
    %v = llvm.load %a {alignment = 4 : i64} : !llvm.ptr -> i32
    llvm.return %v : i32
  }
}

// -----
module {
  llvm.func @escape() -> i64 {
    %one = llvm.mlir.constant(1 : i32) : i32
    %a = llvm.alloca %one x i32 : (i32) -> !llvm.ptr
    // expected-error @below {{pointer escapes or has an unsupported alias use}}
    %address = llvm.ptrtoint %a : !llvm.ptr to i64
    llvm.return %address : i64
  }
}

// -----
module {
  llvm.func @volatile() -> i32 {
    %one = llvm.mlir.constant(1 : i32) : i32
    %a = llvm.alloca %one x i32 : (i32) -> !llvm.ptr
    // expected-error @below {{volatile and atomic accesses require explicit memory semantics}}
    %v = llvm.load volatile %a : !llvm.ptr -> i32
    llvm.return %v : i32
  }
}

// -----
module {
  llvm.func @unknown_attribute() -> i32 {
    %one = llvm.mlir.constant(1 : i32) : i32
    %a = llvm.alloca %one x i32 : (i32) -> !llvm.ptr
    // expected-error @below {{unproven memory access attribute}}
    %v = llvm.load %a {unproven = true} : !llvm.ptr -> i32
    llvm.return %v : i32
  }
}

// -----
// expected-error @below {{invalid source LLVM data layout}}
module attributes {vrc.llvm_data_layout = "invalid-layout"} {
  llvm.func @bad_layout() {
    llvm.return
  }
}

// -----
module attributes {vrc.llvm_data_layout = "e-p:32:32"} {
  llvm.func @inconsistent_pointer_layout() {
    %one = llvm.mlir.constant(1 : i32) : i32
    // expected-error @below {{source and MLIR pointer layouts disagree}}
    %a = llvm.alloca %one x i32 : (i32) -> !llvm.ptr
    llvm.return
  }
}

// -----
module attributes {
  vrc.llvm_data_layout = "",
  dlti.dl_spec = #dlti.dl_spec<i32 = dense<[64, 64]> : vector<2xi64>>
} {
  llvm.func @padded_mlir_scalar() {
    %one = llvm.mlir.constant(1 : i32) : i32
    // expected-error @below {{MLIR scalar allocation stride has unsupported padding}}
    %a = llvm.alloca %one x i32 : (i32) -> !llvm.ptr
    llvm.return
  }
}

// -----
// expected-error @below {{conflicting captured and explicit LLVM data layouts}}
module attributes {vrc.llvm_data_layout = "", llvm.data_layout = "e-p:32:32"} {
  llvm.func @conflicting_layouts() {
    llvm.return
  }
}
