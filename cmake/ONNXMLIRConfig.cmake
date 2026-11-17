include(CMakeFindDependencyMacro)
find_dependency(MLIR CONFIG)

get_filename_component(_onnxmlir_prefix "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
set(_onnxmlir_component_names
  OMONNXOps
  OMONNXElementsAttr
  OMResultTypeInferenceOpInterface
  OMShapeInferenceOpInterface
  OMShapeHelperOpInterface
  OMHasOnnxSubgraphOpInterface
  OMSpecializedKernelOpInterface
  OMMlirDialects
  OMCompilerOptions
  OMCompilerDialects
  OMInitAccelerators
  OMKrnlOps
  OMAccelerator
  OMSupport
  OMMlirUtilities
  OMSmallFPConversion
  OMDiagnostic
  OMBuilder
  OMShapeInference
  OMShapeInferencePass
  OMONNXRewrite
  OMONNXSimplifyShapeRelatedOps
  OMONNXStandardFuncReturnPass
  onnx
  libprotobuf)
set(_onnxmlir_archives)
foreach(_component IN LISTS _onnxmlir_component_names)
  if(_component STREQUAL "libprotobuf")
    if(WIN32)
      set(_archive "${_onnxmlir_prefix}/lib/libprotobuf.lib")
    else()
      set(_archive "${_onnxmlir_prefix}/lib/libprotobuf.a")
    endif()
  elseif(WIN32)
    set(_archive "${_onnxmlir_prefix}/lib/${_component}.lib")
  else()
    set(_archive "${_onnxmlir_prefix}/lib/lib${_component}.a")
  endif()
  if(NOT EXISTS "${_archive}")
    message(FATAL_ERROR "ONNX-MLIR SDK is missing ${_archive}")
  endif()
  list(APPEND _onnxmlir_archives "${_archive}")
endforeach()
if(WIN32)
  file(GLOB _onnxmlir_protobuf_support
    "${_onnxmlir_prefix}/lib/absl_*.lib"
    "${_onnxmlir_prefix}/lib/libutf8_*.lib")
else()
  file(GLOB _onnxmlir_protobuf_support
    "${_onnxmlir_prefix}/lib/libabsl_*.a"
    "${_onnxmlir_prefix}/lib/libutf8_*.a")
endif()
list(APPEND _onnxmlir_archives ${_onnxmlir_protobuf_support})
if(NOT _onnxmlir_archives)
  message(FATAL_ERROR "ONNX-MLIR SDK has no static libraries under ${_onnxmlir_prefix}/lib")
endif()

add_library(ONNXMLIR::Frontend INTERFACE IMPORTED)
set_property(TARGET ONNXMLIR::Frontend PROPERTY INTERFACE_INCLUDE_DIRECTORIES
  "${_onnxmlir_prefix}/include/onnx-mlir"
  "${_onnxmlir_prefix}/include/onnx-mlir/include"
  "${_onnxmlir_prefix}/include/onnx-mlir/third_party/onnx"
  "${_onnxmlir_prefix}/include/onnx-mlir/build/third_party/onnx"
  "${_onnxmlir_prefix}/include/onnx-mlir/build/_deps/protobuf-src/src"
  "${_onnxmlir_prefix}/include/onnx-mlir/build/_deps/absl-src")
set_property(TARGET ONNXMLIR::Frontend PROPERTY INTERFACE_COMPILE_DEFINITIONS
  "ONNX_ML=1;ONNX_NAMESPACE=onnx")
if(WIN32)
  set_property(TARGET ONNXMLIR::Frontend PROPERTY INTERFACE_LINK_LIBRARIES
    "${_onnxmlir_archives};MLIRAffineDialect;MLIRMathDialect;MLIRShapeDialect;MLIRVectorDialect;MLIRLLVMIRTransforms;MLIROpenMPToLLVMIRTranslation")
else()
  list(JOIN _onnxmlir_archives "," _onnxmlir_archive_group)
  set_property(TARGET ONNXMLIR::Frontend PROPERTY INTERFACE_LINK_LIBRARIES
    "$<LINK_GROUP:RESCAN,${_onnxmlir_archive_group}>;MLIRAffineDialect;MLIRMathDialect;MLIRShapeDialect;MLIRVectorDialect;MLIRLLVMIRTransforms;MLIROpenMPToLLVMIRTranslation")
endif()

add_library(ONNXMLIR::Dialect INTERFACE IMPORTED)
set_property(TARGET ONNXMLIR::Dialect PROPERTY INTERFACE_LINK_LIBRARIES
  ONNXMLIR::Frontend)

unset(_onnxmlir_archive_group)
unset(_onnxmlir_archives)
unset(_onnxmlir_protobuf_support)
unset(_onnxmlir_component_names)
unset(_archive)
unset(_component)
unset(_onnxmlir_prefix)
