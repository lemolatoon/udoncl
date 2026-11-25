cmake_minimum_required(VERSION 3.24)
project(udoncl_onnx_user_kernel_test LANGUAGES NONE)

# An ONNX Host plus an OpenCL C kernel of its own in one Unity library: the
# Host C (host.c) runs the MNIST model through the UdonONNX session API and
# launches the ArgMax of argmax.cl over the logits, with its tensors in
# cl_mem arrays created, bound and released in loops.
find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)
find_package(UdonCL 0.1 CONFIG REQUIRED)
find_package(UdonBLAS 0.1 CONFIG REQUIRED)
find_package(UdonDNN 0.1 CONFIG REQUIRED)
find_package(UdonATen 0.1 CONFIG REQUIRED)
find_package(UdonONNXOperators 0.1 CONFIG REQUIRED)
find_package(UdonONNXRuntime 0.1 CONFIG REQUIRED)

set(UDONCL_MNIST_ONNX_GENERATOR "" CACHE FILEPATH
  "examples/mnist-cnn/generate_onnx.py of the source tree")
set(model "${CMAKE_CURRENT_BINARY_DIR}/mnist.onnx")
execute_process(
  COMMAND "${Python3_EXECUTABLE}" "${UDONCL_MNIST_ONNX_GENERATOR}"
          -o "${model}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "MNIST ONNX generation failed:\n${stdout}${stderr}")
endif()

udoncl_add_onnx_host_library(user_kernel_model_host MODEL "${model}"
  DIMENSIONS batch=1)
target_include_directories(user_kernel_model_host INTERFACE
  "${UdonBLAS_INCLUDE_DIRS}"
  "${UdonDNN_INCLUDE_DIRS}"
  "${UdonATen_INCLUDE_DIRS}")
udoncl_add_host_library(user_kernel_onnx_host "${CMAKE_CURRENT_SOURCE_DIR}/host.c")
target_include_directories(user_kernel_onnx_host INTERFACE
  "${UdonONNXRuntime_INCLUDE_DIRS}")
udoncl_add_device_library(user_kernel_argmax
  "${CMAKE_CURRENT_SOURCE_DIR}/argmax.cl")
udoncl_add_unity_library(onnx-user-kernel
  HOST_LIBRARIES
    user_kernel_model_host
    user_kernel_onnx_host
    UdonONNXRuntime::UdonONNXRuntime
    UdonDNN::OpenCLRuntime
  DEVICE_LIBRARIES
    UdonATen::Device
    UdonBLAS::Device
    user_kernel_argmax)
