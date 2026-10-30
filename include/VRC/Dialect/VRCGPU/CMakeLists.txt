set(LLVM_TARGET_DEFINITIONS VRCGPUDialect.td)
mlir_tablegen(VRCGPUDialect.h.inc -gen-dialect-decls -dialect=vrcgpu EXTRA_INCLUDES ${VRC_TABLEGEN_INCLUDES})
mlir_tablegen(VRCGPUDialect.cpp.inc -gen-dialect-defs -dialect=vrcgpu EXTRA_INCLUDES ${VRC_TABLEGEN_INCLUDES})
add_public_tablegen_target(VRCVRCGPUIncGen)
