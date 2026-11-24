// RUN: %vrc-opt --vrc-raise-llvm-raw-memory %s | %FileCheck %s

// A pointer-typed initializer naming an internal global, such as a constant
// table of strings, becomes relocated words of the raw global: the address
// of the global (plus a constant GEP offset into it). The bytes under a
// relocation are zero.
// CHECK: "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, constant, initial_value = array<i8: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>, linkage = "internal", relocations = [{addend = 0 : i32, offset = 0 : i32, symbol = @first}, {addend = 1 : i32, offset = 4 : i32, symbol = @second}], storage_type = !llvm.array<3 x ptr>, sym_name = "names"}>
// CHECK: "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, initial_value = 0 : i32, linkage = "internal", relocations = [{addend = 0 : i32, offset = 0 : i32, symbol = @first}], storage_type = !udonhl.ptr<0>, sym_name = "current"}>
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<32> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, "dlti.endianness" = "little">} {
  llvm.mlir.global internal constant @names() {addr_space = 0 : i32, alignment = 4 : i64} : !llvm.array<3 x ptr> {
    %0 = llvm.mlir.undef : !llvm.array<3 x ptr>
    %1 = llvm.mlir.addressof @first : !llvm.ptr
    %2 = llvm.insertvalue %1, %0[0] : !llvm.array<3 x ptr>
    %3 = llvm.mlir.addressof @second : !llvm.ptr
    %4 = llvm.getelementptr inbounds %3[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i8>
    %5 = llvm.insertvalue %4, %2[1] : !llvm.array<3 x ptr>
    %6 = llvm.mlir.zero : !llvm.ptr
    %7 = llvm.insertvalue %6, %5[2] : !llvm.array<3 x ptr>
    llvm.return %7 : !llvm.array<3 x ptr>
  }
  llvm.mlir.global internal @current() {addr_space = 0 : i32, alignment = 4 : i64} : !llvm.ptr {
    %0 = llvm.mlir.addressof @first : !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.mlir.global private constant @first("a\00") {addr_space = 0 : i32, alignment = 1 : i64}
  llvm.mlir.global private constant @second("xyz\00") {addr_space = 0 : i32, alignment = 1 : i64}
}
