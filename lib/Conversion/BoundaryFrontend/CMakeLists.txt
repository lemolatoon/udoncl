add_mlir_library(VRCBoundaryFrontend DISABLE_INSTALL
  BoundaryFrontend.cpp GatherDevice.cpp
  DEPENDS VRCBoundaryPassIncGen
  LINK_LIBS PUBLIC VRCLLVMToHighLevelMLIR VRCDialectVRC
    MLIRMathDialect MLIRGPUDialect MLIRMemRefDialect MLIRArithDialect MLIRFuncDialect MLIRTransforms
)
vrc_configure_target(VRCBoundaryFrontend)
