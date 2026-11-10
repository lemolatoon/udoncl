add_mlir_library(VRCConversionUdon DISABLE_INSTALL LowerScalar.cpp HostStorageLiveness.cpp
  DEPENDS VRCBoundaryPassIncGen
  LINK_LIBS PUBLIC VRCManifest VRCDialectUdonAsm MLIRArithDialect MLIRFuncDialect
    MLIRMemRefDialect MLIRPass)
vrc_configure_target(VRCConversionUdon)
