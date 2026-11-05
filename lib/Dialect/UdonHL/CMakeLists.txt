add_mlir_dialect_library(VRCDialectUdonHL DISABLE_INSTALL UdonHLDialect.cpp
  DEPENDS VRCUdonHLIncGen
  LINK_LIBS PUBLIC VRCDialectVRC MLIRIR MLIRFuncDialect)
vrc_configure_target(VRCDialectUdonHL)
