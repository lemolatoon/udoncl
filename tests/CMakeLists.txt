find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)
add_test(NAME dialect-roundtrip
  COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/dialects/check.py"
    $<TARGET_FILE:vrc-opt>
)
add_test(NAME llvm-import
  COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/recovery/check.py"
    $<TARGET_FILE:vrc-translate> $<TARGET_FILE:vrc-opt>
)

# Use tools from the exact SDK that provided the linked libraries, never PATH.
function(vrc_find_test_tool variable tool)
  # Do not retain a tool path from an SDK selected by an earlier configure.
  find_program(vrc_sdk_tool NAMES "${tool}"
    HINTS "${LLVM_TOOLS_BINARY_DIR}" NO_DEFAULT_PATH NO_CACHE REQUIRED)
  set(${variable} "${vrc_sdk_tool}" PARENT_SCOPE)
endfunction()
vrc_find_test_tool(VRC_TEST_CLANG clang)
vrc_find_test_tool(VRC_TEST_LLVM_OPT opt)
vrc_find_test_tool(VRC_TEST_MLIR_OPT mlir-opt)
vrc_find_test_tool(VRC_TEST_MLIR_TRANSLATE mlir-translate)
add_test(NAME opencl-frontend-metadata
  COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/frontend/opencl.py"
    $<TARGET_FILE:vrc-translate> $<TARGET_FILE:vrc-opt>
    "${VRC_TEST_CLANG}" "${VRC_TEST_LLVM_OPT}"
    "${PROJECT_BINARY_DIR}/tests/opencl"
)
add_test(NAME llvm-high-level-recovery
  COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/recovery/raise.py"
    $<TARGET_FILE:vrc-translate> $<TARGET_FILE:vrc-opt>
    "${VRC_TEST_MLIR_OPT}" "${VRC_TEST_MLIR_TRANSLATE}" "${VRC_TEST_CLANG}"
    "${PROJECT_BINARY_DIR}/tests/recovery"
)
add_test(NAME llvm-recovery-diagnostics
  COMMAND vrc-opt --vrc-raise-llvm-to-scalar --split-input-file --verify-diagnostics
    "${CMAKE_CURRENT_SOURCE_DIR}/recovery/invalid.mlir"
)
add_test(NAME llvm-memory-recovery
  COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/recovery/memory.py"
    $<TARGET_FILE:vrc-translate> $<TARGET_FILE:vrc-opt>
    "${VRC_TEST_MLIR_OPT}" "${VRC_TEST_MLIR_TRANSLATE}" "${VRC_TEST_CLANG}"
    "${PROJECT_BINARY_DIR}/tests/memory"
)
add_test(NAME llvm-memory-diagnostics
  COMMAND vrc-opt --vrc-raise-llvm-memory --split-input-file --verify-diagnostics
    "${CMAKE_CURRENT_SOURCE_DIR}/recovery/memory-invalid.mlir"
)
add_test(NAME host-device-boundary
  COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/boundary/check.py"
    $<TARGET_FILE:vrc-opt> $<TARGET_FILE:vrccl>
    "${PROJECT_BINARY_DIR}/tests/boundary"
)
add_test(NAME host-device-boundary-diagnostics
  COMMAND vrc-opt --split-input-file --verify-diagnostics
    "${CMAKE_CURRENT_SOURCE_DIR}/boundary/invalid.mlir"
)
add_test(NAME c-boundary-example
  COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/boundary/c_example.py"
    $<TARGET_FILE:vrccl> $<TARGET_FILE:vrc-translate> $<TARGET_FILE:vrc-opt>
    "${VRC_TEST_MLIR_OPT}" "${VRC_TEST_MLIR_TRANSLATE}" "${VRC_TEST_CLANG}"
    "${PROJECT_BINARY_DIR}" "${PROJECT_BINARY_DIR}/tests/c-boundary"
)
add_test(NAME prebuilt-sdk-installer
  COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/build/install_llvm.py"
)
add_test(NAME build-parallelism
  COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/build/parallel.py"
)

if(VRC_ENABLE_NATIVE_FRAGMENT_TESTS)
  if(NOT WIN32)
    find_package(Vulkan REQUIRED)
    find_program(VRC_GLSLANG_VALIDATOR glslangValidator REQUIRED)
  endif()
  add_test(NAME native-fragment-execution
    COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/boundary/native_fragment.py"
      "${PROJECT_BINARY_DIR}" "${PROJECT_BINARY_DIR}/tests/native-fragment")
  set_tests_properties(native-fragment-execution PROPERTIES TIMEOUT 600)
  add_test(NAME opencl-host-execution
    COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/boundary/opencl_host.py"
      "${PROJECT_BINARY_DIR}" "${PROJECT_BINARY_DIR}/tests/opencl-host")
  set_tests_properties(opencl-host-execution PROPERTIES TIMEOUT 600)
endif()
