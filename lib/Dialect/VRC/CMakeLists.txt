add_mlir_dialect_library(VRCDialectVRC DISABLE_INSTALL VRCDialect.cpp
  DEPENDS VRCVRCIncGen
  LINK_LIBS PUBLIC MLIRIR)
vrc_configure_target(VRCDialectVRC)
