add_mlir_library(VRCConversionHost DISABLE_INSTALL HighLevelToUdonHL.cpp
  DEPENDS VRCBoundaryPassIncGen
  LINK_LIBS PUBLIC VRCDialectUdonHL VRCDialectVRC MLIRFuncDialect
    MLIRArithDialect MLIRPass MLIRSCFDialect MLIRSCFToControlFlow
    MLIRTransformUtils MLIRUBDialect)
vrc_configure_target(VRCConversionHost)
