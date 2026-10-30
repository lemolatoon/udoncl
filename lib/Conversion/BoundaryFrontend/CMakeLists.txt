add_mlir_library(VRCBoundaryFrontend DISABLE_INSTALL
  BoundaryFrontend.cpp
  DEPENDS VRCBoundaryPassIncGen
  LINK_LIBS PUBLIC VRCLLVMToHighLevelMLIR VRCDialectVRC
    MLIRGPUDialect MLIRMemRefDialect MLIRArithDialect MLIRFuncDialect MLIRTransforms
)
vrc_configure_target(VRCBoundaryFrontend)
