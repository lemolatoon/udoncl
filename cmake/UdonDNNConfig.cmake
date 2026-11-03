include_guard(GLOBAL)
include(CMakeFindDependencyMacro)

find_dependency(UdonCL 0.1 CONFIG)

get_filename_component(_UDONDNN_CONFIG_DIR "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY)
get_filename_component(_UDONDNN_INSTALL_PREFIX "${_UDONDNN_CONFIG_DIR}/../../.." ABSOLUTE)
get_filename_component(_UDONDNN_SOURCE_HINT "${_UDONDNN_CONFIG_DIR}/.." ABSOLUTE)

if(EXISTS "${_UDONDNN_INSTALL_PREFIX}/share/udondnn/src/udondnn.c")
  set(_UDONDNN_ROOT "${_UDONDNN_INSTALL_PREFIX}/share/udondnn")
  set(_UDONATEN_ROOT "${_UDONDNN_INSTALL_PREFIX}/share/udonaten")
  set(_UDONDNN_INCLUDE_DIR "${_UDONDNN_INSTALL_PREFIX}/include")
  set(_UDONATEN_INCLUDE_DIR "${_UDONDNN_INSTALL_PREFIX}/include")
else()
  set(_UDONDNN_ROOT "${_UDONDNN_SOURCE_HINT}/libraries/udondnn")
  set(_UDONATEN_ROOT "${_UDONDNN_SOURCE_HINT}/libraries/udonaten")
  set(_UDONDNN_INCLUDE_DIR "${_UDONDNN_ROOT}/include")
  set(_UDONATEN_INCLUDE_DIR "${_UDONATEN_ROOT}/include")
endif()

if(NOT TARGET udondnn_host)
  file(GLOB _UDONDNN_DEVICE_SOURCES CONFIGURE_DEPENDS
    "${_UDONDNN_ROOT}/device/*.cl"
    "${_UDONATEN_ROOT}/device/*.cl")
  list(SORT _UDONDNN_DEVICE_SOURCES)
  set(_UDONDNN_DEVICE_SOURCE "")
  foreach(_UDONDNN_DEVICE_FILE IN LISTS _UDONDNN_DEVICE_SOURCES)
    file(READ "${_UDONDNN_DEVICE_FILE}" _UDONDNN_DEVICE_TEXT)
    string(APPEND _UDONDNN_DEVICE_SOURCE "${_UDONDNN_DEVICE_TEXT}\n")
  endforeach()
  string(REPLACE "\\" "\\\\" _UDONDNN_DEVICE_SOURCE
    "${_UDONDNN_DEVICE_SOURCE}")
  string(REPLACE "\"" "\\\"" _UDONDNN_DEVICE_SOURCE
    "${_UDONDNN_DEVICE_SOURCE}")
  string(REPLACE "\n" "\\n" _UDONDNN_DEVICE_SOURCE
    "${_UDONDNN_DEVICE_SOURCE}")
  set(_UDONDNN_GENERATED_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/udondnn-generated")
  file(MAKE_DIRECTORY "${_UDONDNN_GENERATED_DIR}")
  file(WRITE "${_UDONDNN_GENERATED_DIR}/udondnn_device_source.h"
    "#define UDONDNN_DEVICE_SOURCE \"${_UDONDNN_DEVICE_SOURCE}\"\n")

  udoncl_add_host_library(udondnn_host
    "${_UDONDNN_ROOT}/src/udondnn.c")
  target_include_directories(udondnn_host INTERFACE
    "${_UDONDNN_INCLUDE_DIR}"
    "${_UDONATEN_INCLUDE_DIR}"
    "${_UDONDNN_GENERATED_DIR}")
endif()

if(NOT TARGET udondnn_device)
  file(GLOB _UDONDNN_DEVICE_SOURCES CONFIGURE_DEPENDS
    "${_UDONDNN_ROOT}/device/*.cl"
    "${_UDONATEN_ROOT}/device/*.cl")
  list(SORT _UDONDNN_DEVICE_SOURCES)
  udoncl_add_device_library(udondnn_device ${_UDONDNN_DEVICE_SOURCES})
endif()

if(NOT TARGET udondnn)
  add_library(udondnn INTERFACE)
  set_property(TARGET udondnn PROPERTY UDONCL_HOST_LIBRARIES
    udondnn_host UdonCL::OpenCLRuntime)
  set_property(TARGET udondnn PROPERTY UDONCL_DEVICE_LIBRARIES udondnn_device)
endif()
if(NOT TARGET UdonDNN::UdonDNN)
  add_library(UdonDNN::UdonDNN ALIAS udondnn)
endif()

set(UdonDNN_VERSION 0.1.0)
set(UdonDNN_INCLUDE_DIRS "${_UDONDNN_INCLUDE_DIR}")
set(UdonDNN_FOUND TRUE)
