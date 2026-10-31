cmake_minimum_required(VERSION 3.30...4.2)
project(udonblas_device_validation LANGUAGES C CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../lib/Runtime/OpenCL" opencl-runtime)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../lib/Testing/NativeFragment" fragment-runtime)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../lib/Testing/NativeOpenCL" native-opencl-runtime)
add_executable(udonblas-device tests/device_main.cpp
  "${UDONBLAS_HOST_OBJECT}" "${UDONBLAS_DEVICE_OBJECT}")
set_source_files_properties("${UDONBLAS_HOST_OBJECT}" "${UDONBLAS_DEVICE_OBJECT}"
  PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
target_include_directories(udonblas-device PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include" "${UDONBLAS_BINDINGS_DIR}")
target_compile_features(udonblas-device PRIVATE cxx_std_20)
set_target_properties(udonblas-device PROPERTIES CXX_EXTENSIONS OFF)
target_link_libraries(udonblas-device PRIVATE VRCNativeOpenCLRuntime)
