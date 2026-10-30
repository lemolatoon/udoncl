add_mlir_library(VRCLLVMToHighLevelMLIR DISABLE_INSTALL
  Import.cpp
  FrontendMetadata.cpp
  RaiseLLVMToScalar.cpp
  DEPENDS VRCRecoveryPassIncGen
  LINK_LIBS PUBLIC
    MLIRArithDialect
    MLIRControlFlowDialect
    MLIRControlFlowToSCF
    MLIRDLTIDialect
    MLIRFuncDialect
    MLIRLLVMDialect
    MLIRLLVMIRToLLVMTranslation
    MLIRPass
    MLIRSCFDialect
    MLIRTargetLLVMIRImport
    MLIRTransforms
    MLIRTransformUtils
)
vrc_configure_target(VRCLLVMToHighLevelMLIR)
