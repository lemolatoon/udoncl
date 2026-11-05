add_mlir_library(VRCManifest DISABLE_INSTALL BindingSource.cpp Boundary.cpp UdonSharpABI.cpp
  DEPENDS VRCBoundaryPassIncGen
  LINK_LIBS PUBLIC VRCDialectUdonHL VRCDialectVRC VRCDialectVRCGPU
    MLIRFuncDialect MLIRArithDialect MLIRPass)
vrc_configure_target(VRCManifest)
