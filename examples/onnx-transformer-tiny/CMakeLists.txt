cmake_minimum_required(VERSION 3.24)
project(udoncl_onnx_transformer_tiny LANGUAGES NONE)

find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)
find_package(UdonCL 0.1 CONFIG REQUIRED)
find_package(UdonBLAS 0.1 CONFIG REQUIRED)
find_package(UdonDNN 0.1 CONFIG REQUIRED)
find_package(UdonATen 0.1 CONFIG REQUIRED)
find_package(UdonONNXOperators 0.1 CONFIG REQUIRED)
find_package(UdonONNXRuntime 0.1 CONFIG REQUIRED)

set(UDONCL_TINYLLAMA_ONNX_MODEL "" CACHE FILEPATH
  "Use an external TinyLlama-compatible ONNX model instead of the fast synthetic fixture")
if(UDONCL_TINYLLAMA_ONNX_MODEL)
  cmake_path(ABSOLUTE_PATH UDONCL_TINYLLAMA_ONNX_MODEL NORMALIZE
             OUTPUT_VARIABLE model)
  if(NOT EXISTS "${model}")
    message(FATAL_ERROR "TinyLlama ONNX model does not exist: ${model}")
  endif()
  set(model_shape_arguments
    DIMENSION_BOUNDS
      batch_size=1
      sequence_length=4
      past_sequence_length=0
    INPUT_SHAPES
      attention_mask=1,4)
else()
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
  set(model_shape_arguments DIMENSION_BOUNDS sequence=4)
endif()

udoncl_add_onnx_host_library(transformer_host MODEL "${model}"
  ${model_shape_arguments})
target_include_directories(transformer_host INTERFACE
  "${UdonBLAS_INCLUDE_DIRS}"
  "${UdonDNN_INCLUDE_DIRS}"
  "${UdonATen_INCLUDE_DIRS}")
udoncl_add_host_library(transformer_user_host host.c)
target_include_directories(transformer_user_host INTERFACE
  "${UdonONNXRuntime_INCLUDE_DIRS}")
set(unity_library_arguments)
if(UDONCL_TINYLLAMA_ONNX_MODEL)
  list(APPEND unity_library_arguments OMIT_INTERMEDIATE_IR)
endif()
udoncl_add_unity_library(onnx-transformer-tiny
  CONFIG "${CMAKE_CURRENT_SOURCE_DIR}/unity-library.json"
  ${unity_library_arguments}
  HOST_LIBRARIES
    transformer_host
    transformer_user_host
    UdonONNXRuntime::UdonONNXRuntime
  LIBRARIES
    UdonONNXOperators::UdonONNXOperators
    UdonATen::UdonATen
    UdonBLAS::UdonBLAS)
