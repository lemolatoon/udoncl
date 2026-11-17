add_llvm_executable(udoncl-onnx-frontend onnx-frontend.cpp)
target_sources(udoncl-onnx-frontend PRIVATE ${VRC_ONNX_LOCAL_PASS_SOURCES})
llvm_update_compile_flags(udoncl-onnx-frontend)
target_link_libraries(udoncl-onnx-frontend PRIVATE
  VRCONNXToUdonHost
  VRCONNXHostMetadata
  VRCDialectUdonHL
  ONNXMLIR::Frontend
  MLIROptLib
  MLIRRegisterAllPasses
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions)
vrc_configure_target(udoncl-onnx-frontend)
if(MSVC)
  target_compile_options(udoncl-onnx-frontend PRIVATE /GR /EHsc)
else()
  target_compile_options(udoncl-onnx-frontend PRIVATE -frtti -fexceptions)
endif()
