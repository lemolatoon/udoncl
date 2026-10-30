add_mlir_library(VRCManifest DISABLE_INSTALL Boundary.cpp
  DEPENDS VRCBoundaryPassIncGen
  LINK_LIBS PUBLIC VRCDialectUdonHL VRCDialectVRC VRCDialectVRCGPU
    MLIRFuncDialect MLIRArithDialect MLIRPass)
vrc_configure_target(VRCManifest)
