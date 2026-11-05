cmake_minimum_required(VERSION 3.24)
project(udoncl_onnx_mnist LANGUAGES NONE)

find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)
find_package(UdonCL 0.1 CONFIG REQUIRED)
find_package(UdonBLAS 0.1 CONFIG REQUIRED)
find_package(UdonDNN 0.1 CONFIG REQUIRED)
find_package(UdonATen 0.1 CONFIG REQUIRED)
find_package(UdonONNXOperators 0.1 CONFIG REQUIRED)
find_package(UdonONNXRuntime 0.1 CONFIG REQUIRED)

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

udoncl_add_onnx_host_library(mnist_host MODEL "${model}"
  DIMENSION_BOUNDS batch=1)
target_include_directories(mnist_host INTERFACE
  "${UdonBLAS_INCLUDE_DIRS}"
  "${UdonDNN_INCLUDE_DIRS}"
  "${UdonATen_INCLUDE_DIRS}")
udoncl_add_host_library(mnist_user_host host.c)
target_include_directories(mnist_user_host INTERFACE
  "${UdonONNXRuntime_INCLUDE_DIRS}")
udoncl_add_unity_library(onnx-mnist
  CONFIG "${CMAKE_CURRENT_SOURCE_DIR}/unity-library.json"
  HOST_LIBRARIES
    mnist_host
    mnist_user_host
    UdonONNXRuntime::UdonONNXRuntime
  LIBRARIES
    UdonATen::UdonATen
    UdonBLAS::UdonBLAS)
