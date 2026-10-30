add_mlir_dialect_library(VRCDialectVRCGPU DISABLE_INSTALL VRCGPUDialect.cpp
  DEPENDS VRCVRCGPUIncGen
  LINK_LIBS PUBLIC MLIRIR MLIRArithDialect)
vrc_configure_target(VRCDialectVRCGPU)
