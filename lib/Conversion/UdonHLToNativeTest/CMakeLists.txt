add_mlir_library(VRCNativeTestLowering DISABLE_INSTALL
  UdonHLToNativeTest.cpp NativeHostCalls.cpp
  DEPENDS VRCBoundaryPassIncGen
  LINK_LIBS PUBLIC VRCManifest VRCDialectUdonHL VRCDialectVRC VRCDialectVRCGPU
    MLIRLLVMDialect MLIRFuncDialect MLIRFuncTransforms MLIRArithDialect MLIRTransforms
)
vrc_configure_target(VRCNativeTestLowering)
