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
  "${PROJECT_SOURCE_DIR}/cmake/llvm-lock.json"
  "${PROJECT_SOURCE_DIR}/cmake/onnx-mlir-lock.json")
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
add_subdirectory(lib/Analysis)
add_subdirectory(lib/Conversion/LLVMToHighLevel)
add_subdirectory(lib/Conversion/BoundaryFrontend)
add_subdirectory(lib/Conversion/ONNXToUdonHost)
add_subdirectory(lib/Conversion/ONNXHostMetadata)
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
install(TARGETS vrccl vrc-opt RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        COMPONENT UdonCLCompiler)
install(FILES
  cmake/UdonCLConfig.cmake
  cmake/UdonCLConfigVersion.cmake
  cmake/UdonCLCompileArtifact.cmake
  cmake/UdonCLCompileUnityLibrary.cmake
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/UdonCL"
  COMPONENT UdonCLCompiler)
install(FILES
  cmake/UdonBLASConfig.cmake
  cmake/UdonBLASConfigVersion.cmake
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/UdonBLAS"
  COMPONENT UdonCLCompiler)
install(FILES
  cmake/UdonDNNConfig.cmake
  cmake/UdonDNNConfigVersion.cmake
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/UdonDNN"
  COMPONENT UdonCLCompiler)
install(FILES
  cmake/UdonATenConfig.cmake
  cmake/UdonATenConfigVersion.cmake
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/UdonATen"
  COMPONENT UdonCLCompiler)
install(FILES
  cmake/UdonONNXOperatorsConfig.cmake
  cmake/UdonONNXOperatorsConfigVersion.cmake
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/UdonONNXOperators"
  COMPONENT UdonCLCompiler)
install(FILES
  cmake/UdonONNXRuntimeConfig.cmake
  cmake/UdonONNXRuntimeConfigVersion.cmake
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/UdonONNXRuntime"
  COMPONENT UdonCLCompiler)
install(FILES
  cmake/UdonTokenizersConfig.cmake
  cmake/UdonTokenizersConfigVersion.cmake
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/UdonTokenizers"
  COMPONENT UdonCLCompiler)
install(FILES
  include/udoncl/export.h
  include/udoncl/array.h
  include/udoncl/string.h
  include/udoncl/unity_buffer.h
  include/udoncl/device_memory.h
  include/udoncl/onnx_binding_abi.h
  include/udoncl/onnx_generated_binding.h
  include/udoncl/onnx_types.h
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/udoncl"
  COMPONENT UdonCLCompiler)
install(FILES include/VRC/Runtime/OpenCL/Backend.h
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/VRC/Runtime/OpenCL"
  COMPONENT UdonCLCompiler)
install(FILES
  lib/Runtime/OpenCL/OpenCLAOT.c
  lib/Runtime/OpenCL/DeviceMemory.c
  DESTINATION "${CMAKE_INSTALL_DATADIR}/udoncl/runtime"
  COMPONENT UdonCLCompiler)
install(FILES
  runtime/onnx/udon_onnx.py
  runtime/onnx/udon_pages.py
  runtime/onnx/udon_weights.py
  scripts/compile_onnx_mlir.py
  DESTINATION "${CMAKE_INSTALL_DATADIR}/udoncl/onnx"
  COMPONENT UdonCLCompiler)

unset(VRC_TEST_ONNX_IMPORTER CACHE)
unset(VRC_TEST_ONNX_OPT CACHE)
find_program(VRC_TEST_ONNX_IMPORTER NAMES udoncl-onnx
  HINTS
    "${PROJECT_SOURCE_DIR}/.deps/onnx-mlir/bin"
  NO_DEFAULT_PATH)
find_program(VRC_TEST_ONNX_OPT NAMES udoncl-onnx-opt
  HINTS
    "${PROJECT_SOURCE_DIR}/.deps/onnx-mlir/bin"
  NO_DEFAULT_PATH)
install(FILES libraries/udonblas/include/udonblas.h
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  COMPONENT UdonCLCompiler)
install(FILES libraries/udonblas/src/udonblas.c
  DESTINATION "${CMAKE_INSTALL_DATADIR}/udonblas/src"
  COMPONENT UdonCLCompiler)
install(DIRECTORY libraries/udonblas/device
  DESTINATION "${CMAKE_INSTALL_DATADIR}/udonblas"
  COMPONENT UdonCLCompiler
  FILES_MATCHING PATTERN "*.cl")
install(FILES libraries/udondnn/include/udondnn.h
  libraries/udonaten/include/udonaten.h
  libraries/udononnxoperators/include/udononnxoperators.h
  libraries/udononnxruntime/include/udononnxruntime.h
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  COMPONENT UdonCLCompiler)
install(FILES libraries/udontokenizers/include/udontokenizers.h
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  COMPONENT UdonCLCompiler)
install(FILES libraries/udontokenizers/src/udontokenizers.c
  DESTINATION "${CMAKE_INSTALL_DATADIR}/udontokenizers/src"
  COMPONENT UdonCLCompiler)
install(FILES scripts/compile_tokenizer_tables.py
  DESTINATION "${CMAKE_INSTALL_DATADIR}/udontokenizers"
  COMPONENT UdonCLCompiler)
install(FILES libraries/udondnn/src/udondnn.c
  DESTINATION "${CMAKE_INSTALL_DATADIR}/udondnn/src"
  COMPONENT UdonCLCompiler)
install(FILES libraries/udonaten/src/udonaten.c
  DESTINATION "${CMAKE_INSTALL_DATADIR}/udonaten/src"
  COMPONENT UdonCLCompiler)
install(FILES libraries/udononnxoperators/src/matmul_nbits.c
  DESTINATION "${CMAKE_INSTALL_DATADIR}/udononnxoperators/src"
  COMPONENT UdonCLCompiler)
install(FILES libraries/udononnxruntime/src/udononnxruntime.c
              libraries/udononnxruntime/src/onnx_binding_abi.c
  DESTINATION "${CMAKE_INSTALL_DATADIR}/udononnxruntime/src"
  COMPONENT UdonCLCompiler)
install(DIRECTORY libraries/udononnxoperators/device
  DESTINATION "${CMAKE_INSTALL_DATADIR}/udononnxoperators"
  COMPONENT UdonCLCompiler
  FILES_MATCHING PATTERN "*.cl")
install(DIRECTORY libraries/udondnn/device
  DESTINATION "${CMAKE_INSTALL_DATADIR}/udondnn"
  COMPONENT UdonCLCompiler
  FILES_MATCHING PATTERN "*.cl")
install(DIRECTORY libraries/udonaten/device
  DESTINATION "${CMAKE_INSTALL_DATADIR}/udonaten"
  COMPONENT UdonCLCompiler
  FILES_MATCHING PATTERN "*.cl")
install(DIRECTORY third_party/OpenCL-Headers/CL
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  COMPONENT UdonCLCompiler)
install(FILES third_party/OpenCL-Headers/LICENSE
  DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/UdonCL"
  RENAME OpenCL-Headers.LICENSE
  COMPONENT UdonCLCompiler)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
