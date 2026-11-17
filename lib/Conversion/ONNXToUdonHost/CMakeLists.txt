add_mlir_library(VRCONNXToUdonHost DISABLE_INSTALL
  FinalizeONNXHost.cpp
  ONNXHostGraph.cpp
  ONNXHostPlan.cpp
  ONNXHostSymbols.cpp
  ONNXToUdonHost.cpp
  ONNXToUdonHostCore.cpp
  PARTIAL_SOURCES_INTENDED
  DEPENDS VRCBoundaryPassIncGen
  LINK_LIBS PUBLIC
    MLIRArithDialect
    MLIRControlFlowDialect
    MLIRFuncDialect
    MLIRIR
    MLIRLLVMDialect
    MLIRPass
    VRCDialectUdonHL
)
vrc_configure_target(VRCONNXToUdonHost)
