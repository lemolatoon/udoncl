cmake_minimum_required(VERSION 3.30...4.2)
project(udondnn_device_validation LANGUAGES C CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../lib/Runtime/OpenCL" opencl-runtime)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../lib/Testing/NativeFragment" fragment-runtime)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../lib/Testing/NativeOpenCL" native-opencl-runtime)
add_executable(udondnn-device tests/device_main.cpp
  "${UDONDNN_HOST_OBJECT}" "${UDONATEN_OBJECT}" "${UDONDNN_DEVICE_OBJECT}"
  "${UDONDNN_ACTIVATION_OBJECT}" "${UDONDNN_POOLING_OBJECT}"
  "${UDONDNN_SOFTMAX_OBJECT}" "${UDONDNN_BATCHNORM_OBJECT}"
  "${UDONDNN_OP_TENSOR_OBJECT}")
target_sources(udondnn-device PRIVATE "${UDONDNN_RMS_NORM_OBJECT}")
target_sources(udondnn-device PRIVATE "${UDONDNN_GATHER_OBJECT}")
target_sources(udondnn-device PRIVATE "${UDONATEN_INT4_OBJECT}")
target_sources(udondnn-device PRIVATE "${UDONATEN_TILE_OBJECT}")
set_source_files_properties("${UDONDNN_HOST_OBJECT}" "${UDONDNN_DEVICE_OBJECT}"
  "${UDONATEN_OBJECT}"
  "${UDONDNN_ACTIVATION_OBJECT}" "${UDONDNN_POOLING_OBJECT}"
  "${UDONDNN_SOFTMAX_OBJECT}" "${UDONDNN_BATCHNORM_OBJECT}"
  "${UDONDNN_OP_TENSOR_OBJECT}"
  "${UDONDNN_RMS_NORM_OBJECT}"
  "${UDONDNN_GATHER_OBJECT}"
  "${UDONATEN_INT4_OBJECT}"
  "${UDONATEN_TILE_OBJECT}"
  PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
target_include_directories(udondnn-device PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/../udonaten/include"
  "${UDONDNN_BINDINGS_DIR}")
target_compile_features(udondnn-device PRIVATE cxx_std_20)
set_target_properties(udondnn-device PROPERTIES CXX_EXTENSIONS OFF)
target_link_libraries(udondnn-device PRIVATE VRCNativeOpenCLRuntime)
