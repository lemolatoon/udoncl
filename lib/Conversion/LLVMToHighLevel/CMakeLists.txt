add_mlir_library(VRCLLVMToHighLevelMLIR DISABLE_INSTALL
  Import.cpp
  FrontendMetadata.cpp
  RaiseLLVMToScalar.cpp
  RaiseLLVMMemory.cpp
  RaiseLLVMRawMemory.cpp
  DEPENDS VRCRecoveryPassIncGen
  LINK_LIBS PUBLIC
    VRCDialectVRC
    VRCDialectUdonHL
    MLIRArithDialect
    MLIRControlFlowDialect
    MLIRControlFlowToSCF
    MLIRDLTIDialect
    MLIRFuncDialect
    MLIRLLVMDialect
    MLIRLLVMIRToLLVMTranslation
    MLIRMathDialect
    MLIRGPUDialect
    MLIRMemRefDialect
    MLIRPass
    MLIRSCFDialect
    MLIRTargetLLVMIRImport
    MLIRTransforms
    MLIRTransformUtils
)
vrc_configure_target(VRCLLVMToHighLevelMLIR)
