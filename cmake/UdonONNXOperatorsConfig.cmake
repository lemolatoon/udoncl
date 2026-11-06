include_guard(GLOBAL)
include(CMakeFindDependencyMacro)

find_dependency(UdonCL 0.1 CONFIG)

get_filename_component(_UDONONNX_CONFIG_DIR "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY)
get_filename_component(_UDONONNX_INSTALL_PREFIX "${_UDONONNX_CONFIG_DIR}/../../.." ABSOLUTE)
get_filename_component(_UDONONNX_SOURCE_HINT "${_UDONONNX_CONFIG_DIR}/.." ABSOLUTE)

if(EXISTS "${_UDONONNX_INSTALL_PREFIX}/share/udononnxoperators/src/matmul_nbits.c")
  set(_UDONONNX_ROOT "${_UDONONNX_INSTALL_PREFIX}/share/udononnxoperators")
  set(_UDONONNX_INCLUDE_DIR "${_UDONONNX_INSTALL_PREFIX}/include")
else()
  set(_UDONONNX_ROOT "${_UDONONNX_SOURCE_HINT}/libraries/udononnxoperators")
  set(_UDONONNX_INCLUDE_DIR "${_UDONONNX_ROOT}/include")
endif()

if(NOT TARGET udononnxoperators_host)
  set(_UDONONNX_DEVICE_FILES
    "${_UDONONNX_ROOT}/device/matmul_nbits.cl"
    "${_UDONONNX_ROOT}/device/gather.cl"
    "${_UDONONNX_ROOT}/device/unary.cl")
  set(_UDONONNX_DEVICE_SOURCE "")
  foreach(_UDONONNX_DEVICE_FILE IN LISTS _UDONONNX_DEVICE_FILES)
    file(READ "${_UDONONNX_DEVICE_FILE}" _UDONONNX_DEVICE_PART)
    string(APPEND _UDONONNX_DEVICE_SOURCE
      "${_UDONONNX_DEVICE_PART}\n")
  endforeach()
  string(REPLACE "\\" "\\\\" _UDONONNX_DEVICE_SOURCE
    "${_UDONONNX_DEVICE_SOURCE}")
  string(REPLACE "\"" "\\\"" _UDONONNX_DEVICE_SOURCE
    "${_UDONONNX_DEVICE_SOURCE}")
  string(REPLACE "\n" "\\n" _UDONONNX_DEVICE_SOURCE
    "${_UDONONNX_DEVICE_SOURCE}")
  set(_UDONONNX_GENERATED_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/udononnxoperators-generated")
  file(MAKE_DIRECTORY "${_UDONONNX_GENERATED_DIR}")
  file(WRITE
    "${_UDONONNX_GENERATED_DIR}/udononnxoperators_device_source.h"
    "#define UDONONNX_DEVICE_SOURCE \"${_UDONONNX_DEVICE_SOURCE}\"\n")
  udoncl_add_host_library(udononnxoperators_host
    "${_UDONONNX_ROOT}/src/matmul_nbits.c")
  target_include_directories(udononnxoperators_host INTERFACE
    "${_UDONONNX_INCLUDE_DIR}"
    "${_UDONONNX_GENERATED_DIR}")
endif()

if(NOT TARGET udononnxoperators_device)
  udoncl_add_device_library(udononnxoperators_device
    ${_UDONONNX_DEVICE_FILES})
endif()

if(NOT TARGET udononnxoperators_opencl_runtime)
  if(EXISTS "${_UDONONNX_INSTALL_PREFIX}/share/udoncl/runtime/OpenCLAOT.c")
    set(_UDONONNX_OPENCL_RUNTIME
      "${_UDONONNX_INSTALL_PREFIX}/share/udoncl/runtime/OpenCLAOT.c")
  else()
    set(_UDONONNX_OPENCL_RUNTIME
      "${_UDONONNX_SOURCE_HINT}/lib/Runtime/OpenCL/OpenCLAOT.c")
  endif()
  udoncl_add_host_library(udononnxoperators_opencl_runtime
    "${_UDONONNX_OPENCL_RUNTIME}")
  target_compile_definitions(udononnxoperators_opencl_runtime INTERFACE
    CL_TARGET_OPENCL_VERSION=120
    UDONCL_AOT_ENABLE_DNN=1
    UDONCL_AOT_ENABLE_ONNX=1)
endif()

if(NOT TARGET udononnxoperators)
  add_library(udononnxoperators INTERFACE)
  set_property(TARGET udononnxoperators PROPERTY UDONCL_HOST_LIBRARIES
    udononnxoperators_host udononnxoperators_opencl_runtime)
  set_property(TARGET udononnxoperators PROPERTY UDONCL_DEVICE_LIBRARIES
    udononnxoperators_device)
endif()
if(NOT TARGET UdonONNXOperators::UdonONNXOperators)
  add_library(UdonONNXOperators::UdonONNXOperators ALIAS udononnxoperators)
endif()

set(UdonONNXOperators_VERSION 0.1.0)
set(UdonONNXOperators_INCLUDE_DIRS "${_UDONONNX_INCLUDE_DIR}")
set(UdonONNXOperators_FOUND TRUE)
