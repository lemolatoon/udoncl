add_mlir_library(VRCTargetHLSL DISABLE_INSTALL Emitter.cpp GatherEmitter.cpp
  WordTexelEmitter.cpp
  LINK_LIBS PUBLIC VRCDialectVRCGPU MLIRSCFDialect MLIRMathDialect MLIRArithDialect)
vrc_configure_target(VRCTargetHLSL)
