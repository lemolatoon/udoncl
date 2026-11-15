add_mlir_library(VRCConversionUdon DISABLE_INSTALL LowerScalar.cpp
  DEPENDS VRCBoundaryPassIncGen
  LINK_LIBS PUBLIC VRCAnalysisHostStorage VRCManifest VRCDialectUdonAsm
    MLIRArithDialect MLIRFuncDialect MLIRMemRefDialect MLIRPass)
vrc_configure_target(VRCConversionUdon)
