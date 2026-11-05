add_mlir_dialect_library(VRCDialectVRC DISABLE_INSTALL VRCDialect.cpp
  DEPENDS VRCVRCIncGen
  LINK_LIBS PUBLIC MLIRIR MLIRAsyncDialect MLIRFuncDialect)
vrc_configure_target(VRCDialectVRC)
