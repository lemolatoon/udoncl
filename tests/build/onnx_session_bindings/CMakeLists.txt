cmake_minimum_required(VERSION 3.24)
project(udoncl_onnx_session_bindings_test LANGUAGES NONE)

# A Unity library whose Host binds UDONCL_PUBLIC buffers to the graph inputs
# and output of its model by name (host.c) through the UdonONNX session API.
find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)
find_package(UdonCL 0.1 CONFIG REQUIRED)
find_package(UdonONNXOperators 0.1 CONFIG REQUIRED)
find_package(UdonONNXRuntime 0.1 CONFIG REQUIRED)

set(model "${CMAKE_CURRENT_BINARY_DIR}/session-bindings.onnx")
execute_process(
  COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/generate_onnx.py" -o "${model}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "ONNX generation failed:\n${stdout}${stderr}")
endif()

udoncl_add_onnx_host_library(session_model_host MODEL "${model}")
target_include_directories(session_model_host INTERFACE
  "${UdonONNXOperators_INCLUDE_DIRS}")
# The test builds variants of the Host from a copy of host.c.
set(UDONCL_SESSION_BINDINGS_HOST "${CMAKE_CURRENT_SOURCE_DIR}/host.c"
  CACHE FILEPATH "Host C source of the library")
udoncl_add_host_library(session_user_host "${UDONCL_SESSION_BINDINGS_HOST}")
target_include_directories(session_user_host INTERFACE
  "${UdonONNXRuntime_INCLUDE_DIRS}")
udoncl_add_unity_library(onnx-session-bindings
  HOST_LIBRARIES
    session_model_host
    session_user_host
    UdonONNXRuntime::UdonONNXRuntime
    UdonCL::OpenCLRuntime)

# A model with a bounded dimension: present = Concat(past, step), bound to
# the leading blocks of fixed buffers with the run-time extent of each run
# (bounded_host.c).
set(bounded_model "${CMAKE_CURRENT_BINARY_DIR}/session-bindings-bounded.onnx")
execute_process(
  COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/generate_onnx.py" --bounded
          -o "${bounded_model}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "ONNX generation failed:\n${stdout}${stderr}")
endif()
udoncl_add_onnx_host_library(session_bounded_model_host
  MODEL "${bounded_model}"
  DIMENSION_BOUNDS past=15)
target_include_directories(session_bounded_model_host INTERFACE
  "${UdonONNXOperators_INCLUDE_DIRS}")
set(UDONCL_SESSION_BINDINGS_BOUNDED_HOST
  "${CMAKE_CURRENT_SOURCE_DIR}/bounded_host.c"
  CACHE FILEPATH "Host C source of the bounded library")
udoncl_add_host_library(session_bounded_user_host
  "${UDONCL_SESSION_BINDINGS_BOUNDED_HOST}")
target_include_directories(session_bounded_user_host INTERFACE
  "${UdonONNXRuntime_INCLUDE_DIRS}")
udoncl_add_unity_library(onnx-session-bindings-bounded
  HOST_LIBRARIES
    session_bounded_model_host
    session_bounded_user_host
    UdonONNXRuntime::UdonONNXRuntime
    UdonCL::OpenCLRuntime)
