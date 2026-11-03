#ifndef VRC_LLVM_TO_HIGH_LEVEL_PASSES
#define VRC_LLVM_TO_HIGH_LEVEL_PASSES
include "mlir/Pass/PassBase.td"

def RaiseLLVMToScalarPass : Pass<"vrc-raise-llvm-to-scalar", "mlir::ModuleOp"> {
  let summary = "Raise proven scalar LLVM functions and CFG to func/arith/cf";
  let description = [{
    Fully converts the supported scalar subset of LLVM dialect. Preserves SSA
    block arguments, branch operands, source locations, and nsw/nuw flags.
    Unsupported memory, pointer, poison, ABI or instruction semantics fail with
    a diagnostic rather than being left in a supposedly recovered module.
    This is the first stage of comprehensive high-level recovery; it does not
    claim memref, affine, GPU or barrier/dependence recovery.
  }];
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::cf::ControlFlowDialect",
                           "mlir::func::FuncDialect"];
}
def RaiseLLVMMemoryPass : Pass<"vrc-raise-llvm-memory", "mlir::ModuleOp"> {
  let summary = "Recover proven nonescaping fixed-size local memory as memref";
  let description = [{
    Recover entry-block AS0 allocas of scalar/array types as flattened static
    memrefs. Trace GEP provenance, validate layout, offsets, access types,
    alignment and definite initialization before rewriting. Recognize constant,
    masked and guarded counted-loop indices. Reject unsupported escape, alias,
    volatile, atomic and unproven memory semantics. Other LLVM operations remain
    for the scalar conversion stage; no pointer casts are used as a fallback.
  }];
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::memref::MemRefDialect"];
}
def RaiseLLVMRawMemoryPass : Pass<"vrc-raise-llvm-raw-memory", "mlir::ModuleOp"> {
  let summary = "Recover C pointers and byte-addressed memory as UdonHL";
  let description = [{
    Convert remaining LLVM pointer types, raw allocas, GEPs, loads, stores,
    pointer-valued direct calls and simple globals to the UdonHL logical byte
    memory model. This runs after typed memref recovery so proven homogeneous
    arrays retain their higher-level representation.
  }];
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::func::FuncDialect",
                           "vrc::udonhl::UdonHLDialect"];
}
#endif
