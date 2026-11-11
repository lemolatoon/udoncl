if(TARGET OMONNXOps)
  target_sources(onnx-mlir-opt PRIVATE
    CanonicalizeONNXLibraryPatterns.cpp
    ExternalizeONNXInitializers.cpp
    IdentifyONNXSequenceState.cpp
    InferONNXShapes.cpp
    MaterializeONNXConstants.cpp
    ONNXToUdonHostCore.cpp
    ONNXToUdonHostStandalone.cpp
  )
  add_custom_target(udoncl-onnx-opt
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_FILE:onnx-mlir-opt>
      $<TARGET_FILE_DIR:onnx-mlir-opt>/udoncl-onnx-opt${CMAKE_EXECUTABLE_SUFFIX}
    DEPENDS onnx-mlir-opt
  )
  return()
endif()

add_mlir_library(VRCONNXToUdonHost DISABLE_INSTALL
  FinalizeONNXHost.cpp
  ONNXToUdonHost.cpp
  ONNXToUdonHostCore.cpp
  PARTIAL_SOURCES_INTENDED
  DEPENDS VRCBoundaryPassIncGen
  LINK_LIBS PUBLIC
    MLIRArithDialect
    MLIRControlFlowDialect
    MLIRFuncDialect
    MLIRIR
    MLIRLLVMDialect
    MLIRPass
    VRCDialectUdonHL
)
vrc_configure_target(VRCONNXToUdonHost)
