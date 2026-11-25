# Build hooks for the locked ONNX-MLIR SDK build.
#
# scripts/install_onnx_mlir.py passes this file as
# CMAKE_PROJECT_onnx-mlir_INCLUDE, so it runs right after project(onnx-mlir)
# and before ONNX-MLIR loads LLVM/MLIR or declares any of its own targets.

# Placeholder targets for LLVM test utilities named by ONNX-MLIR's lit suites.
# ONNX-MLIR declares its check-* suites even with ONNX_MLIR_BUILD_TESTS=OFF,
# and they add target dependencies on FileCheck, count, not and split-file.
# The official Linux LLVM release archive does not export these tools, so
# generation fails there. The installer lists only the tools the locked SDK
# does not export. UdonCL never builds the check-* targets, so a placeholder
# fails loudly if anything does build it.
foreach(tool IN LISTS UDONCL_ONNX_MLIR_LIT_TOOL_STUBS)
  if(NOT TARGET "${tool}")
    add_custom_target("${tool}"
      COMMAND "${CMAKE_COMMAND}" -E echo
              "${tool} is not exported by the locked LLVM SDK"
      COMMAND "${CMAKE_COMMAND}" -E false
      VERBATIM)
  endif()
endforeach()

# ONNX-MLIR libraries include generated pass headers without depending on the
# TableGen target that writes them. For example, OMONNXOps compiles
# ONNXDimAnalysis.cpp, which includes src/Pass/Passes.hpp and through it
# src/Transform/PassesKrnl.h.inc, but only OMKrnlOps and OMLowerKrnlRegion
# depend on OMTransformsKrnlPassIncGen. Building the ONNX-MLIR tools happened
# to order the generator first; building only the SDK libraries does not.
# Collect every public TableGen target into one umbrella target, which the
# installer builds before the libraries. LLVM's add_public_tablegen_target
# files these targets in a ".../Tablegenning" folder; ONNX-MLIR's custom
# *IncGen targets that rewrite its sources are not public TableGen targets.
function(udoncl_onnx_mlir_collect_tablegen directory output)
  set(result)
  get_property(targets DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
  foreach(target IN LISTS targets)
    get_property(folder TARGET "${target}" PROPERTY FOLDER)
    if(folder MATCHES "/Tablegenning$")
      list(APPEND result "${target}")
    endif()
  endforeach()
  get_property(subdirectories DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)
  foreach(subdirectory IN LISTS subdirectories)
    udoncl_onnx_mlir_collect_tablegen("${subdirectory}" nested)
    list(APPEND result ${nested})
  endforeach()
  set(${output} "${result}" PARENT_SCOPE)
endfunction()

function(udoncl_onnx_mlir_add_tablegen_target)
  udoncl_onnx_mlir_collect_tablegen("${CMAKE_SOURCE_DIR}" targets)
  if(NOT targets)
    message(FATAL_ERROR "ONNX-MLIR declared no public TableGen targets")
  endif()
  add_custom_target(udoncl-onnx-mlir-tablegen)
  add_dependencies(udoncl-onnx-mlir-tablegen ${targets})
endfunction()

# Run after ONNX-MLIR's top-level CMakeLists.txt has declared every target.
cmake_language(DEFER CALL udoncl_onnx_mlir_add_tablegen_target)
