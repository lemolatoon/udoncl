include_guard(GLOBAL)
include(CMakeFindDependencyMacro)

find_dependency(UdonCL 0.1 CONFIG)

get_filename_component(_UDONTOKENIZERS_CONFIG_DIR "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY)
get_filename_component(_UDONTOKENIZERS_INSTALL_PREFIX "${_UDONTOKENIZERS_CONFIG_DIR}/../../.." ABSOLUTE)
get_filename_component(_UDONTOKENIZERS_SOURCE_HINT "${_UDONTOKENIZERS_CONFIG_DIR}/.." ABSOLUTE)

if(EXISTS "${_UDONTOKENIZERS_INSTALL_PREFIX}/share/udontokenizers/src/udontokenizers.c")
  set(_UDONTOKENIZERS_ROOT "${_UDONTOKENIZERS_INSTALL_PREFIX}/share/udontokenizers")
  set(_UDONTOKENIZERS_INCLUDE_DIR "${_UDONTOKENIZERS_INSTALL_PREFIX}/include")
else()
  set(_UDONTOKENIZERS_ROOT "${_UDONTOKENIZERS_SOURCE_HINT}/libraries/udontokenizers")
  set(_UDONTOKENIZERS_INCLUDE_DIR "${_UDONTOKENIZERS_ROOT}/include")
endif()

if(NOT TARGET udontokenizers_host)
  udoncl_add_host_library(udontokenizers_host
    "${_UDONTOKENIZERS_ROOT}/src/udontokenizers.c")
  target_include_directories(udontokenizers_host INTERFACE
    "${_UDONTOKENIZERS_INCLUDE_DIR}")
endif()
if(NOT TARGET udontokenizers)
  add_library(udontokenizers INTERFACE)
  set_property(TARGET udontokenizers PROPERTY UDONCL_HOST_LIBRARIES
    udontokenizers_host)
endif()
if(NOT TARGET UdonTokenizers::UdonTokenizers)
  add_library(UdonTokenizers::UdonTokenizers ALIAS udontokenizers)
endif()

set(UdonTokenizers_VERSION 0.1.0)
set(UdonTokenizers_INCLUDE_DIRS "${_UDONTOKENIZERS_INCLUDE_DIR}")
set(UdonTokenizers_FOUND TRUE)
