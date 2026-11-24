add_mlir_library(VRCONNXToUdonHost DISABLE_INSTALL
  AttachONNXHostMetadata.cpp
  CleanupONNXHostSource.cpp
  FinalizeONNXHost.cpp
  MaterializeONNXHostGlobals.cpp
  MaterializeONNXHostModelRegistry.cpp
  ONNXHostFusion.cpp
  ONNXHostFusionKernel.cpp
  ONNXHostGraph.cpp
  ONNXHostPlan.cpp
  ONNXDimensionExprs.cpp
  ONNXHostSymbols.cpp
  ONNXToUdonHost.cpp
  ONNXToUdonHostCore.cpp
  PARTIAL_SOURCES_INTENDED
  DEPENDS VRCBoundaryPassIncGen
  LINK_LIBS PUBLIC
    MLIRArithDialect
    MLIRControlFlowDialect
    MLIRFuncDialect
    MLIRGPUDialect
    MLIRIR
    MLIRLLVMDialect
    MLIRMathDialect
    MLIRMemRefDialect
    MLIRPass
    MLIRSCFDialect
    VRCDialectUdonHL
)
vrc_configure_target(VRCONNXToUdonHost)
