include_guard(GLOBAL)
include(CMakeFindDependencyMacro)

find_dependency(UdonCL 0.1 CONFIG)

get_filename_component(_UDONONNX_CONFIG_DIR "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY)
get_filename_component(_UDONONNX_INSTALL_PREFIX "${_UDONONNX_CONFIG_DIR}/../../.." ABSOLUTE)
get_filename_component(_UDONONNX_SOURCE_HINT "${_UDONONNX_CONFIG_DIR}/.." ABSOLUTE)

if(EXISTS "${_UDONONNX_INSTALL_PREFIX}/share/udononnxoperators/src/udononnxoperators.c")
  set(_UDONONNX_ROOT "${_UDONONNX_INSTALL_PREFIX}/share/udononnxoperators")
  set(_UDONONNX_INCLUDE_DIR "${_UDONONNX_INSTALL_PREFIX}/include")
else()
  set(_UDONONNX_ROOT "${_UDONONNX_SOURCE_HINT}/libraries/udononnxoperators")
  set(_UDONONNX_INCLUDE_DIR "${_UDONONNX_ROOT}/include")
endif()

if(NOT TARGET udononnxoperators_host)
  udoncl_add_host_library(udononnxoperators_host
    "${_UDONONNX_ROOT}/src/udononnxoperators.c")
  target_include_directories(udononnxoperators_host INTERFACE
    "${_UDONONNX_INCLUDE_DIR}")
endif()

if(NOT TARGET udononnxoperators)
  add_library(udononnxoperators INTERFACE)
  set_property(TARGET udononnxoperators PROPERTY UDONCL_HOST_LIBRARIES
    udononnxoperators_host UdonCL::OpenCLRuntime)
endif()
if(NOT TARGET UdonONNXOperators::UdonONNXOperators)
  add_library(UdonONNXOperators::UdonONNXOperators ALIAS udononnxoperators)
endif()

set(UdonONNXOperators_VERSION 0.1.0)
set(UdonONNXOperators_INCLUDE_DIRS "${_UDONONNX_INCLUDE_DIR}")
set(UdonONNXOperators_FOUND TRUE)
