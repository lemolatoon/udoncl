add_mlir_library(VRCNativeTestLowering DISABLE_INSTALL
  UdonHLToNativeTest.cpp
  DEPENDS VRCBoundaryPassIncGen
  LINK_LIBS PUBLIC VRCManifest VRCDialectUdonHL VRCDialectVRC VRCDialectVRCGPU
    MLIRFuncDialect MLIRFuncTransforms MLIRArithDialect MLIRTransforms
)
vrc_configure_target(VRCNativeTestLowering)
