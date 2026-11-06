include_guard(GLOBAL)
include(CMakeFindDependencyMacro)

find_dependency(UdonDNN 0.1 CONFIG)

get_filename_component(_UDONATEN_CONFIG_DIR "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY)
get_filename_component(_UDONATEN_INSTALL_PREFIX "${_UDONATEN_CONFIG_DIR}/../../.." ABSOLUTE)
get_filename_component(_UDONATEN_SOURCE_HINT "${_UDONATEN_CONFIG_DIR}/.." ABSOLUTE)

if(EXISTS "${_UDONATEN_INSTALL_PREFIX}/share/udonaten/src/udonaten.c")
  set(_UDONATEN_ROOT "${_UDONATEN_INSTALL_PREFIX}/share/udonaten")
  set(_UDONATEN_INCLUDE_DIR "${_UDONATEN_INSTALL_PREFIX}/include")
else()
  set(_UDONATEN_ROOT "${_UDONATEN_SOURCE_HINT}/libraries/udonaten")
  set(_UDONATEN_INCLUDE_DIR "${_UDONATEN_ROOT}/include")
endif()

if(NOT TARGET udonaten_host)
  udoncl_add_host_library(udonaten_host
    "${_UDONATEN_ROOT}/src/udonaten.c")
  target_include_directories(udonaten_host INTERFACE
    "${_UDONATEN_INCLUDE_DIR}"
    "${UdonDNN_INCLUDE_DIRS}")
  target_compile_definitions(udonaten_host INTERFACE
    UDONATEN_AOT_DIRECT_LARGE=1)
endif()

if(NOT TARGET udonaten)
  add_library(udonaten INTERFACE)
  set_property(TARGET udonaten PROPERTY UDONCL_HOST_LIBRARIES
    udonaten_host UdonDNN::UdonDNN)
  set_property(TARGET udonaten PROPERTY UDONCL_DEVICE_LIBRARIES
    UdonDNN::UdonDNN)
endif()
if(NOT TARGET UdonATen::UdonATen)
  add_library(UdonATen::UdonATen ALIAS udonaten)
endif()
if(NOT TARGET UdonATen::Device)
  add_library(UdonATen::Device ALIAS udondnn_device)
endif()

set(UdonATen_VERSION 0.1.0)
set(UdonATen_INCLUDE_DIRS "${_UDONATEN_INCLUDE_DIR}")
set(UdonATen_FOUND TRUE)
