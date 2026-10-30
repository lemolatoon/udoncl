set(LLVM_TARGET_DEFINITIONS UdonHLDialect.td)
mlir_tablegen(UdonHLDialect.h.inc -gen-dialect-decls -dialect=udonhl EXTRA_INCLUDES ${VRC_TABLEGEN_INCLUDES})
mlir_tablegen(UdonHLDialect.cpp.inc -gen-dialect-defs -dialect=udonhl EXTRA_INCLUDES ${VRC_TABLEGEN_INCLUDES})
add_public_tablegen_target(VRCUdonHLIncGen)
