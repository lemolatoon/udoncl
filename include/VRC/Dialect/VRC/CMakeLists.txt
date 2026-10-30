set(LLVM_TARGET_DEFINITIONS VRCDialect.td)
mlir_tablegen(VRCDialect.h.inc -gen-dialect-decls -dialect=vrc EXTRA_INCLUDES ${VRC_TABLEGEN_INCLUDES})
mlir_tablegen(VRCDialect.cpp.inc -gen-dialect-defs -dialect=vrc EXTRA_INCLUDES ${VRC_TABLEGEN_INCLUDES})
add_public_tablegen_target(VRCVRCIncGen)
