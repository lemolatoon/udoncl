// RUN: %vrc-opt --vrc-raise-llvm-raw-memory %s | %FileCheck %s

// SPIR spells the null of a non-generic address space as a cast of the
// generic null: a file-scope udon_string_t starts as that null.
// CHECK: "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, initial_value = 0 : i32, linkage = "internal", storage_type = !udonhl.ptr<200>, sym_name = "message"}>
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<32> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, "dlti.endianness" = "little">} {
  llvm.mlir.global internal @message() {addr_space = 0 : i32, alignment = 4 : i64} : !llvm.ptr<200> {
    %0 = llvm.mlir.zero : !llvm.ptr<4>
    %1 = llvm.addrspacecast %0 : !llvm.ptr<4> to !llvm.ptr<200>
    llvm.return %1 : !llvm.ptr<200>
  }
}
