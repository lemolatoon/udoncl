add_mlir_library(VRCAnalysisHostStorage DISABLE_INSTALL
  HostStorageLiveness.cpp
  LINK_LIBS PUBLIC VRCDialectUdonHL VRCDialectVRC MLIRArithDialect
    MLIRAnalysis MLIRControlFlowDialect MLIRFuncDialect MLIRIR
    MLIRMemRefDialect)
vrc_configure_target(VRCAnalysisHostStorage)
