add_mlir_library(VRCAnalysisHostStorage DISABLE_INSTALL
  HostGlobalAccesses.cpp
  HostStorageAllocation.cpp
  HostStorageLiveness.cpp
  HostEvaluator.cpp
  LINK_LIBS PUBLIC VRCDialectUdonHL VRCDialectVRC MLIRArithDialect
    MLIRAnalysis MLIRControlFlowDialect MLIRFuncDialect MLIRIR
    MLIRMemRefDialect)
vrc_configure_target(VRCAnalysisHostStorage)
