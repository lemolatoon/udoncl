cmake_minimum_required(VERSION 3.30...4.2)
project(udoncl_onnx_mnist_dense LANGUAGES NONE)

find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)
find_package(UdonCL 0.1 REQUIRED CONFIG)
find_package(UdonBLAS 0.1 REQUIRED CONFIG)

set(model "${CMAKE_CURRENT_BINARY_DIR}/mnist.onnx")
execute_process(
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/generate_onnx.py"
          -o "${model}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "MNIST ONNX generation failed:\n${stdout}${stderr}")
endif()

udoncl_add_onnx_host_library(mnist_dense_host
  MODEL "${model}"
  PROFILE mnist-dense)
target_include_directories(mnist_dense_host INTERFACE "${UdonBLAS_INCLUDE_DIRS}")

udoncl_add_unity_library(onnx-mnist-dense
  CONFIG unity-library.json
  HOST_LIBRARIES mnist_dense_host
  LIBRARIES UdonBLAS::UdonBLAS)
