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
