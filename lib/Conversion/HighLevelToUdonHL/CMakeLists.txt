add_mlir_library(VRCConversionHost DISABLE_INSTALL HighLevelToUdonHL.cpp
  DEPENDS VRCBoundaryPassIncGen
  LINK_LIBS PUBLIC VRCDialectUdonHL VRCDialectVRC MLIRFuncDialect
    MLIRArithDialect MLIRPass MLIRTransformUtils)
vrc_configure_target(VRCConversionHost)
