cmake_minimum_required(VERSION 3.30...4.2)
project(udoncl_mnist_cnn LANGUAGES C CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../lib/Runtime/OpenCL" opencl-runtime)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../lib/Testing/NativeFragment" fragment-runtime)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../lib/Testing/NativeOpenCL" native-opencl-runtime)
add_executable(mnist-cnn-device main.cpp mnist_cnn.c
  "${UDONCL_HOST_OBJECT}" "${UDONCL_DEVICE_OBJECT}")
set_source_files_properties("${UDONCL_HOST_OBJECT}" "${UDONCL_DEVICE_OBJECT}"
  PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
target_include_directories(mnist-cnn-device PRIVATE "${UDONCL_BINDINGS_DIR}")
target_compile_features(mnist-cnn-device PRIVATE cxx_std_20)
set_target_properties(mnist-cnn-device PROPERTIES CXX_EXTENSIONS OFF)
target_link_libraries(mnist-cnn-device PRIVATE VRCNativeOpenCLRuntime)
