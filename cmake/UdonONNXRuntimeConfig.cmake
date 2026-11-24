include_guard(GLOBAL)
include(CMakeFindDependencyMacro)

find_dependency(UdonCL 0.1 CONFIG)

get_filename_component(_UDONONNXRUNTIME_CONFIG_DIR
  "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY)
get_filename_component(_UDONONNXRUNTIME_INSTALL_PREFIX
  "${_UDONONNXRUNTIME_CONFIG_DIR}/../../.." ABSOLUTE)
get_filename_component(_UDONONNXRUNTIME_SOURCE_HINT
  "${_UDONONNXRUNTIME_CONFIG_DIR}/.." ABSOLUTE)

if(EXISTS
   "${_UDONONNXRUNTIME_INSTALL_PREFIX}/share/udononnxruntime/src/udononnxruntime.c")
  set(_UDONONNXRUNTIME_ROOT
    "${_UDONONNXRUNTIME_INSTALL_PREFIX}/share/udononnxruntime")
  set(_UDONONNXRUNTIME_INCLUDE_DIR
    "${_UDONONNXRUNTIME_INSTALL_PREFIX}/include")
  set(_UDONONNXRUNTIME_DEVICE_MEMORY
    "${_UDONONNXRUNTIME_INSTALL_PREFIX}/share/udoncl/runtime/DeviceMemory.c")
else()
  set(_UDONONNXRUNTIME_ROOT
    "${_UDONONNXRUNTIME_SOURCE_HINT}/libraries/udononnxruntime")
  set(_UDONONNXRUNTIME_INCLUDE_DIR
    "${_UDONONNXRUNTIME_ROOT}/include")
  set(_UDONONNXRUNTIME_DEVICE_MEMORY
    "${_UDONONNXRUNTIME_SOURCE_HINT}/lib/Runtime/OpenCL/DeviceMemory.c")
endif()

if(NOT TARGET udononnxruntime_host)
  udoncl_add_host_library(udononnxruntime_host
    "${_UDONONNXRUNTIME_ROOT}/src/udononnxruntime.c"
    "${_UDONONNXRUNTIME_ROOT}/src/onnx_binding_abi.c"
    "${_UDONONNXRUNTIME_DEVICE_MEMORY}")
  target_compile_definitions(udononnxruntime_host INTERFACE
    CL_TARGET_OPENCL_VERSION=120)
  target_include_directories(udononnxruntime_host INTERFACE
    "${_UDONONNXRUNTIME_INCLUDE_DIR}")
endif()

if(NOT TARGET udononnxruntime)
  add_library(udononnxruntime INTERFACE)
  set_property(TARGET udononnxruntime PROPERTY UDONCL_HOST_LIBRARIES
    udononnxruntime_host)
  set_property(TARGET udononnxruntime PROPERTY UDONCL_DEVICE_LIBRARIES "")
endif()
if(NOT TARGET UdonONNXRuntime::UdonONNXRuntime)
  add_library(UdonONNXRuntime::UdonONNXRuntime ALIAS udononnxruntime)
endif()

set(UdonONNXRuntime_VERSION 0.1.0)
set(UdonONNXRuntime_INCLUDE_DIRS "${_UDONONNXRUNTIME_INCLUDE_DIR}")
# udononnxruntime_internal.h: the model registry, tensor helpers and the
# index-bound legacy session. Not public API; only Hosts that still use the
# legacy session add it.
set(UdonONNXRuntime_INTERNAL_INCLUDE_DIRS "${_UDONONNXRUNTIME_ROOT}/src")
set(UdonONNXRuntime_FOUND TRUE)
