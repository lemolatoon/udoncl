cmake_minimum_required(VERSION 3.24)
project(udoncl_onnx_mnist LANGUAGES NONE)

find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)
find_package(UdonCL CONFIG REQUIRED)

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
cmake_path(SET device_source NORMALIZE
  "${CMAKE_CURRENT_SOURCE_DIR}/../mnist-cnn/device.cl")
udoncl_add_device_library(mnist_device "${device_source}")
udoncl_add_unity_library(onnx-mnist
  CONFIG "${CMAKE_CURRENT_SOURCE_DIR}/unity-library.json"
  HOST_LIBRARIES mnist_host UdonCL::OpenCLMnistRuntime
  DEVICE_LIBRARIES mnist_device)
