# Every UdonCL-specific ONNX pass is compiled directly into the
# repository-owned pass driver.
set(VRC_ONNX_LOCAL_PASS_SOURCES
  "${CMAKE_CURRENT_LIST_DIR}/CanonicalizeONNXLibraryPatterns.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/ExternalizeONNXGraphConstants.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/ExternalizeONNXInitializers.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/FoldONNXRepeatedMatMulOperand.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/IdentifyONNXSequenceState.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/InferONNXShapes.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/MaterializeONNXConstants.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/MergeONNXSiblingMatMulNBits.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/ONNXStaticTensor.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/PlanONNXHostGraph.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/SimplifyONNXEmptyConcat.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/SimplifyONNXViews.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/SpecializeONNXInputs.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/ValidateONNXHostGraph.cpp"
)
