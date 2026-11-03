cmake_minimum_required(VERSION 3.24)
project(udoncl_onnx_mnist LANGUAGES NONE)

find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)
find_package(UdonCL 0.1 CONFIG REQUIRED)
find_package(UdonBLAS 0.1 CONFIG REQUIRED)
find_package(UdonDNN 0.1 CONFIG REQUIRED)

# The generated MNIST host keeps four tensor descriptors alive concurrently and
# one descriptor of every other kind. Keep the public UdonDNN default at 64,
# while specializing this closed AOT program so its descriptor storage fits the
# Udon VM heap.
target_compile_definitions(udondnn_host INTERFACE
  UDONDNN_DESCRIPTOR_SLOTS=4
  UDONDNN_AOT_LARGE_ONLY=1
  UDONDNN_AOT_RAW_INTRINSICS=1)
set_property(TARGET udondnn PROPERTY UDONCL_HOST_LIBRARIES udondnn_host)
target_compile_definitions(udonblas_host INTERFACE
  UDONBLAS_AOT_RAW_INTRINSICS=1)
set_property(TARGET udonblas PROPERTY UDONCL_HOST_LIBRARIES udonblas_host)

set(model "${CMAKE_CURRENT_BINARY_DIR}/mnist.onnx")
execute_process(
  COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/../mnist-cnn/generate_onnx.py"
          -o "${model}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "MNIST ONNX generation failed:\n${stdout}${stderr}")
endif()

udoncl_add_onnx_host_library(mnist_host MODEL "${model}" PROFILE mnist)
udoncl_add_unity_library(onnx-mnist
  CONFIG "${CMAKE_CURRENT_SOURCE_DIR}/unity-library.json"
  HOST_LIBRARIES mnist_host
  LIBRARIES UdonDNN::UdonDNN UdonBLAS::UdonBLAS)
