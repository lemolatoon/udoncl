set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name VRCRecovery
  EXTRA_INCLUDES ${VRC_TABLEGEN_INCLUDES})
mlir_tablegen(Passes.md -gen-pass-doc EXTRA_INCLUDES ${VRC_TABLEGEN_INCLUDES})
add_public_tablegen_target(VRCRecoveryPassIncGen)
