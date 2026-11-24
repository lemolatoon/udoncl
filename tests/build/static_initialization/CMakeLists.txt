cmake_minimum_required(VERSION 3.24)
project(udoncl_static_initialization_test LANGUAGES NONE)

# One transpose component built three ways from host.c: with a plain
# `initialize` export, with `initialize` also an __attribute__((constructor))
# function, and with a constructor that is not an export.
find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)
find_package(UdonCL 0.1 CONFIG REQUIRED)
find_package(UdonONNXOperators 0.1 CONFIG REQUIRED)
find_package(UdonONNXRuntime 0.1 CONFIG REQUIRED)

set(model "${CMAKE_CURRENT_BINARY_DIR}/transpose-component.onnx")
execute_process(
  COMMAND "${Python3_EXECUTABLE}"
          "${UDONCL_TEST_EXAMPLE_DIRECTORY}/generate_onnx.py" -o "${model}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "Transpose component ONNX generation failed:\n${stdout}${stderr}")
endif()

udoncl_add_onnx_host_library(model_host MODEL "${model}")
target_include_directories(model_host INTERFACE
  "${UdonONNXOperators_INCLUDE_DIRS}")

foreach(variant IN ITEMS dynamic static hidden)
  udoncl_add_host_library(${variant}_host
    "${CMAKE_CURRENT_SOURCE_DIR}/host.c")
  target_include_directories(${variant}_host INTERFACE
    "${UdonONNXRuntime_INCLUDE_DIRS}")
endforeach()
# udoncl_add_unity_library reads the compile definitions of its Host
# libraries when it is called.
target_compile_definitions(static_host INTERFACE UDONCL_TEST_CONSTRUCTOR)
target_compile_definitions(hidden_host INTERFACE
  UDONCL_TEST_HIDDEN_CONSTRUCTOR)
foreach(variant IN ITEMS dynamic static hidden)
  udoncl_add_unity_library(${variant}
    HOST_LIBRARIES model_host ${variant}_host UdonONNXRuntime::UdonONNXRuntime
    LIBRARIES UdonONNXOperators::UdonONNXOperators)
endforeach()
