add_mlir_dialect_library(VRCDialectVRCGPU DISABLE_INSTALL VRCGPUDialect.cpp
  DEPENDS VRCVRCGPUIncGen
  LINK_LIBS PUBLIC MLIRIR)
vrc_configure_target(VRCDialectVRCGPU)
