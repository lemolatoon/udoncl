cmake_minimum_required(VERSION 3.24)
project(udoncl_onnx_transpose_component LANGUAGES NONE)

find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)
find_package(UdonCL 0.1 CONFIG REQUIRED)
find_package(UdonONNXOperators 0.1 CONFIG REQUIRED)
find_package(UdonONNXRuntime 0.1 CONFIG REQUIRED)

set(model "${CMAKE_CURRENT_BINARY_DIR}/transpose-component.onnx")
execute_process(
  COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/generate_onnx.py"
          -o "${model}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "Transpose component ONNX generation failed:\n${stdout}${stderr}")
endif()

udoncl_add_onnx_host_library(transpose_component_model_host MODEL "${model}")
target_include_directories(transpose_component_model_host INTERFACE
  "${UdonONNXOperators_INCLUDE_DIRS}")
udoncl_add_host_library(transpose_component_user_host host.c)
target_include_directories(transpose_component_user_host INTERFACE
  "${UdonONNXRuntime_INCLUDE_DIRS}")
udoncl_add_unity_library(onnx-transpose-component
  CONFIG "${CMAKE_CURRENT_SOURCE_DIR}/unity-library.json"
  HOST_LIBRARIES
    transpose_component_model_host
    transpose_component_user_host
    UdonONNXRuntime::UdonONNXRuntime
  LIBRARIES UdonONNXOperators::UdonONNXOperators)
