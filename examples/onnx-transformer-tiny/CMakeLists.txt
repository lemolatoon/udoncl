cmake_minimum_required(VERSION 3.24)
project(udoncl_onnx_transformer_tiny LANGUAGES NONE)

find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)
find_package(UdonCL 0.1 CONFIG REQUIRED)
find_package(UdonBLAS 0.1 CONFIG REQUIRED)
find_package(UdonDNN 0.1 CONFIG REQUIRED)
find_package(UdonATen 0.1 CONFIG REQUIRED)
find_package(UdonONNXOperators 0.1 CONFIG REQUIRED)

set(model "${CMAKE_CURRENT_BINARY_DIR}/transformer.onnx")
execute_process(
  COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/generate_onnx.py"
          -o "${model}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "Transformer ONNX generation failed:\n${stdout}${stderr}")
endif()

udoncl_add_onnx_host_library(transformer_host MODEL "${model}"
  DIMENSION_BOUNDS sequence=4)
target_include_directories(transformer_host INTERFACE
  "${UdonBLAS_INCLUDE_DIRS}"
  "${UdonDNN_INCLUDE_DIRS}"
  "${UdonATen_INCLUDE_DIRS}")
udoncl_add_unity_library(onnx-transformer-tiny
  CONFIG "${CMAKE_CURRENT_SOURCE_DIR}/unity-library.json"
  HOST_LIBRARIES transformer_host
  LIBRARIES
    UdonATen::UdonATen
    UdonBLAS::UdonBLAS)
