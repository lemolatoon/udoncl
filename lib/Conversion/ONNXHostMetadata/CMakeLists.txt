add_mlir_library(VRCONNXHostMetadata DISABLE_INSTALL
  ../ONNXToUdonHost/EmitONNXHostMetadata.cpp
  PARTIAL_SOURCES_INTENDED
  DEPENDS VRCBoundaryPassIncGen
  LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
)
vrc_configure_target(VRCONNXHostMetadata)
