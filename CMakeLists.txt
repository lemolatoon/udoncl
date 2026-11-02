cmake_minimum_required(VERSION 3.30...4.2)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "Use an out-of-source build, e.g. cmake --preset udoncl")
endif()
project(udoncl VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Use the same installation for LLVM and MLIR on every host OS.
if(MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/cmake/llvm-lock.json")
file(READ "${PROJECT_SOURCE_DIR}/cmake/llvm-lock.json" vrc_llvm_lock)
string(JSON VRC_LLVM_VERSION GET "${vrc_llvm_lock}" version)
include(cmake/VRCFindLLVM.cmake)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(CTest)
option(VRC_ENABLE_NATIVE_FRAGMENT_TESTS "Run native Host binaries with software fragment rendering" ON)
include(cmake/VRCBuildOptions.cmake)
add_subdirectory(include/VRC/Dialect)
add_subdirectory(include/VRC/Conversion)
add_subdirectory(lib/Dialect)
add_subdirectory(lib/Conversion/LLVMToHighLevel)
add_subdirectory(lib/Conversion/BoundaryFrontend)
add_subdirectory(lib/Conversion/HighLevelToUdonHL)
add_subdirectory(lib/Conversion/VRCToUdonHL)
add_subdirectory(lib/Conversion/HighLevelToVRCGPU)
add_subdirectory(lib/Conversion/UdonHLToNativeTest)
add_subdirectory(lib/Conversion/UdonHLToUdonAsm)
add_subdirectory(lib/ABI/Manifest)
add_subdirectory(lib/Target/HLSL)
add_subdirectory(lib/Target/UdonAssembly)
if(VRC_ENABLE_NATIVE_FRAGMENT_TESTS)
  add_subdirectory(lib/Runtime/OpenCL)
  add_subdirectory(lib/Testing/NativeFragment)
  add_subdirectory(lib/Testing/NativeOpenCL)
endif()
add_subdirectory(tools/vrc-opt)
add_subdirectory(tools/vrc-translate)
add_subdirectory(tools/vrccl)
include(GNUInstallDirs)
install(TARGETS vrccl RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        COMPONENT UdonCLCompiler)
install(FILES
  cmake/UdonCLConfig.cmake
  cmake/UdonCLConfigVersion.cmake
  cmake/UdonCLCompileArtifact.cmake
  cmake/UdonCLCompileUnityLibrary.cmake
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/UdonCL"
  COMPONENT UdonCLCompiler)
install(FILES include/udoncl/unity_buffer.h
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/udoncl"
  COMPONENT UdonCLCompiler)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
