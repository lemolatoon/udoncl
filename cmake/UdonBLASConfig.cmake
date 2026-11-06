include_guard(GLOBAL)
include(CMakeFindDependencyMacro)

find_dependency(UdonCL 0.1 CONFIG)

get_filename_component(_UDONBLAS_CONFIG_DIR "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY)
get_filename_component(_UDONBLAS_INSTALL_PREFIX "${_UDONBLAS_CONFIG_DIR}/../../.." ABSOLUTE)
get_filename_component(_UDONBLAS_SOURCE_HINT "${_UDONBLAS_CONFIG_DIR}/.." ABSOLUTE)

if(EXISTS "${_UDONBLAS_INSTALL_PREFIX}/share/udonblas/src/udonblas.c")
  set(_UDONBLAS_ROOT "${_UDONBLAS_INSTALL_PREFIX}/share/udonblas")
  set(_UDONBLAS_INCLUDE_DIR "${_UDONBLAS_INSTALL_PREFIX}/include")
else()
  set(_UDONBLAS_ROOT "${_UDONBLAS_SOURCE_HINT}/libraries/udonblas")
  set(_UDONBLAS_INCLUDE_DIR "${_UDONBLAS_ROOT}/include")
endif()

if(NOT TARGET udonblas_host)
  file(GLOB _UDONBLAS_DEVICE_SOURCES CONFIGURE_DEPENDS
    "${_UDONBLAS_ROOT}/device/*.cl")
  list(SORT _UDONBLAS_DEVICE_SOURCES)
  set(_UDONBLAS_DEVICE_SOURCE "")
  foreach(_UDONBLAS_DEVICE_FILE IN LISTS _UDONBLAS_DEVICE_SOURCES)
    file(READ "${_UDONBLAS_DEVICE_FILE}" _UDONBLAS_DEVICE_TEXT)
    string(APPEND _UDONBLAS_DEVICE_SOURCE
      "${_UDONBLAS_DEVICE_TEXT}\n")
  endforeach()
  string(REPLACE "\\" "\\\\" _UDONBLAS_DEVICE_SOURCE
    "${_UDONBLAS_DEVICE_SOURCE}")
  string(REPLACE "\"" "\\\"" _UDONBLAS_DEVICE_SOURCE
    "${_UDONBLAS_DEVICE_SOURCE}")
  string(REPLACE "\n" "\\n" _UDONBLAS_DEVICE_SOURCE
    "${_UDONBLAS_DEVICE_SOURCE}")
  set(_UDONBLAS_GENERATED_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/udonblas-generated")
  file(MAKE_DIRECTORY "${_UDONBLAS_GENERATED_DIR}")
  file(WRITE "${_UDONBLAS_GENERATED_DIR}/udonblas_device_source.h"
    "#define UDONBLAS_DEVICE_SOURCE \"${_UDONBLAS_DEVICE_SOURCE}\"\n")

  udoncl_add_host_library(udonblas_host
    "${_UDONBLAS_ROOT}/src/udonblas.c")
  target_include_directories(udonblas_host INTERFACE
    "${_UDONBLAS_INCLUDE_DIR}"
    "${_UDONBLAS_GENERATED_DIR}")
  target_compile_definitions(udonblas_host INTERFACE
    UDONBLAS_AOT_DIRECT_DISPATCH=1)
endif()

if(NOT TARGET udonblas_device)
  file(GLOB _UDONBLAS_DEVICE_SOURCES CONFIGURE_DEPENDS
    "${_UDONBLAS_ROOT}/device/*.cl")
  list(SORT _UDONBLAS_DEVICE_SOURCES)
  udoncl_add_device_library(udonblas_device ${_UDONBLAS_DEVICE_SOURCES})
endif()

if(NOT TARGET udonblas)
  add_library(udonblas INTERFACE)
  set_property(TARGET udonblas PROPERTY UDONCL_HOST_LIBRARIES
    udonblas_host UdonCL::OpenCLRuntime)
  set_property(TARGET udonblas PROPERTY UDONCL_DEVICE_LIBRARIES udonblas_device)
endif()
if(NOT TARGET UdonBLAS::UdonBLAS)
  add_library(UdonBLAS::UdonBLAS ALIAS udonblas)
endif()
if(NOT TARGET UdonBLAS::Device)
  add_library(UdonBLAS::Device ALIAS udonblas_device)
endif()

set(UdonBLAS_VERSION 0.1.0)
set(UdonBLAS_INCLUDE_DIRS "${_UDONBLAS_INCLUDE_DIR}")
set(UdonBLAS_FOUND TRUE)
