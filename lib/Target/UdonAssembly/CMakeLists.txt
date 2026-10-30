add_mlir_library(VRCTargetUdon DISABLE_INSTALL Emitter.cpp
  LINK_LIBS PUBLIC VRCDialectUdonAsm MLIRIR)
vrc_configure_target(VRCTargetUdon)
