add_mlir_dialect_library(VRCDialectUdonAsm DISABLE_INSTALL UdonAsmDialect.cpp
  DEPENDS VRCUdonAsmIncGen
  LINK_LIBS PUBLIC MLIRIR)
vrc_configure_target(VRCDialectUdonAsm)
