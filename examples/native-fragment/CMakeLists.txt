cmake_minimum_required(VERSION 3.30...4.2)
project(udoncl_native_fragment_probe LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT DEFINED UDONCL_HOST_OBJECT OR NOT DEFINED UDONCL_ORACLE_OBJECT)
  message(FATAL_ERROR "Use run.py to generate the Host and oracle objects first")
endif()
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../lib/Testing/NativeFragment"
  "${CMAKE_CURRENT_BINARY_DIR}/native-runtime")
add_executable(native-fragment main.cpp
  "${UDONCL_HOST_OBJECT}" "${UDONCL_ORACLE_OBJECT}")
set_source_files_properties("${UDONCL_HOST_OBJECT}" "${UDONCL_ORACLE_OBJECT}"
  PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
target_compile_features(native-fragment PRIVATE cxx_std_17)
set_target_properties(native-fragment PROPERTIES CXX_EXTENSIONS OFF)
target_include_directories(native-fragment PRIVATE "${UDONCL_BINDINGS_DIR}")
target_link_libraries(native-fragment PRIVATE VRCNativeFragmentRuntime)
