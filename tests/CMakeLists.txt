find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)
add_executable(opencl-aot-runtime
  runtime/opencl_aot.c
  "${PROJECT_SOURCE_DIR}/lib/Runtime/OpenCL/OpenCLAOT.c")
target_compile_features(opencl-aot-runtime PRIVATE c_std_11)
set_target_properties(opencl-aot-runtime PROPERTIES
  C_STANDARD 11 C_STANDARD_REQUIRED YES C_EXTENSIONS OFF)
target_include_directories(opencl-aot-runtime PRIVATE
  "${PROJECT_SOURCE_DIR}/third_party/OpenCL-Headers")
target_compile_definitions(opencl-aot-runtime PRIVATE
  CL_TARGET_OPENCL_VERSION=120
  UDONCL_AOT_ENABLE_MNIST=1)
add_test(NAME opencl-aot-runtime COMMAND opencl-aot-runtime)
add_executable(udondnn-large-convolution-kernel
  boundary/udondnn_large_convolution.cpp)
target_compile_features(udondnn-large-convolution-kernel PRIVATE cxx_std_20)
set_target_properties(udondnn-large-convolution-kernel PROPERTIES
  CXX_EXTENSIONS OFF)
add_test(NAME udondnn-large-convolution-kernel
  COMMAND udondnn-large-convolution-kernel)
add_executable(udondnn-large-pointwise-kernel
  boundary/udondnn_large_pointwise.cpp)
target_compile_features(udondnn-large-pointwise-kernel PRIVATE cxx_std_20)
set_target_properties(udondnn-large-pointwise-kernel PROPERTIES
  CXX_EXTENSIONS OFF)
add_test(NAME udondnn-large-pointwise-kernel
  COMMAND udondnn-large-pointwise-kernel)
add_executable(udondnn-large-pooling-kernel
  boundary/udondnn_large_pooling.cpp)
target_compile_features(udondnn-large-pooling-kernel PRIVATE cxx_std_20)
set_target_properties(udondnn-large-pooling-kernel PROPERTIES
  CXX_EXTENSIONS OFF)
add_test(NAME udondnn-large-pooling-kernel
  COMMAND udondnn-large-pooling-kernel)
add_test(NAME dialect-roundtrip
  COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/dialects/check.py"
    $<TARGET_FILE:vrc-opt>
)
add_test(NAME udon-scalar-codegen
  COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/codegen/udon.py"
    $<TARGET_FILE:vrc-opt> $<TARGET_FILE:vrc-translate>
    "${PROJECT_BINARY_DIR}/tests/udon-scalar"
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
vrc_find_test_tool(VRC_TEST_LLVM_LINK llvm-link)
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
add_test(NAME large-gather-device-lowering
  COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/boundary/large_gather.py"
    $<TARGET_FILE:vrccl> "${VRC_TEST_CLANG}"
    "${PROJECT_BINARY_DIR}/tests/large-gather"
)
add_test(NAME udondnn-large-pooling-lowering
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/boundary/udondnn_large_pooling.py"
    $<TARGET_FILE:vrccl> "${VRC_TEST_CLANG}"
    "${PROJECT_BINARY_DIR}/tests/udondnn-large-pooling"
)
add_test(NAME udondnn-large-convolution-lowering
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/boundary/udondnn_large_convolution.py"
    $<TARGET_FILE:vrccl> "${VRC_TEST_CLANG}"
    "${PROJECT_BINARY_DIR}/tests/udondnn-large-convolution"
)
add_test(NAME udondnn-large-pointwise-lowering
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/boundary/udondnn_large_pointwise.py"
    $<TARGET_FILE:vrccl> "${VRC_TEST_CLANG}"
    "${PROJECT_BINARY_DIR}/tests/udondnn-large-pointwise"
)
add_test(NAME mnist-cnn-native
  COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/boundary/mnist_cnn.py"
    "${PROJECT_BINARY_DIR}" "${PROJECT_BINARY_DIR}/tests/mnist-cnn"
)
add_test(NAME tiny-llama-native
  COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/boundary/tiny_llama.py"
    "${PROJECT_BINARY_DIR}" "${PROJECT_BINARY_DIR}/tests/tiny-llama"
)
add_test(NAME onnx-runtime-mnist
  COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/boundary/onnx_runtime.py"
    "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}/tests/onnx-runtime-mnist"
)
add_test(NAME onnx-host-mnist-dense
  COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/boundary/onnx_host.py"
    "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}/tests/onnx-host-mnist-dense"
    "${VRC_TEST_CLANG}" "$<TARGET_FILE:vrc-opt>"
)
add_test(NAME onnx-runtime-llama-8b
  COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/boundary/llama_8b_onnx.py"
    "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}/tests/onnx-runtime-llama-8b"
)
add_test(NAME onnx-arena-add-example
  COMMAND Python3::Interpreter "${PROJECT_SOURCE_DIR}/examples/onnx-arena-add/build.py"
    --output "${PROJECT_BINARY_DIR}/tests/onnx-arena-add-example" --plan-only
)
add_test(NAME onnx-tiny-llama-example
  COMMAND Python3::Interpreter "${PROJECT_SOURCE_DIR}/examples/tiny-llama/build_onnx.py"
    --output "${PROJECT_BINARY_DIR}/tests/onnx-tiny-llama-example" --plan-only
)
add_test(NAME onnx-weight-packer
  COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/boundary/onnx_weights.py"
    "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}/tests/onnx-weight-packer"
)
add_test(NAME prebuilt-sdk-installer
  COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/build/install_llvm.py"
)
add_test(NAME onnx-mlir-frontend-installer
  COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/build/install_onnx_mlir.py"
)
add_test(NAME onnx-mlir-pipeline
  COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/build/compile_onnx_mlir.py"
)
add_test(NAME build-parallelism
  COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/build/parallel.py"
)
add_test(NAME cmake-package-consumer
  COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/build/cmake_package.py"
    --source "${PROJECT_SOURCE_DIR}/examples/hello-kernel"
    --output "${PROJECT_BINARY_DIR}/tests/cmake-package"
    --project-build "${PROJECT_BINARY_DIR}"
    --cmake "${CMAKE_COMMAND}"
    --generator "${CMAKE_GENERATOR}"
    --make-program "${CMAKE_MAKE_PROGRAM}"
    --compiler $<TARGET_FILE:vrccl>
    --clang "${VRC_TEST_CLANG}"
    --llvm-link "${VRC_TEST_LLVM_LINK}"
)
set_tests_properties(cmake-package-consumer PROPERTIES TIMEOUT 300)

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
  add_test(NAME md-crystal-execution
    COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/boundary/md_crystal.py"
      "${PROJECT_BINARY_DIR}" "${PROJECT_BINARY_DIR}/tests/md-crystal")
  set_tests_properties(md-crystal-execution PROPERTIES TIMEOUT 600)
  add_test(NAME mnist-cnn-device-execution
    COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/boundary/mnist_cnn_device.py"
      "${PROJECT_BINARY_DIR}" "${PROJECT_BINARY_DIR}/tests/mnist-cnn-device")
  set_tests_properties(mnist-cnn-device-execution PROPERTIES TIMEOUT 600)
  add_test(NAME tiny-llama-device-execution
    COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/boundary/tiny_llama_device.py"
      "${PROJECT_BINARY_DIR}" "${PROJECT_BINARY_DIR}/tests/tiny-llama-device")
  set_tests_properties(tiny-llama-device-execution PROPERTIES TIMEOUT 600)
  add_test(NAME udonblas-device-execution
    COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/boundary/udonblas_device.py"
      "${PROJECT_BINARY_DIR}" "${PROJECT_BINARY_DIR}/tests/udonblas-device")
  set_tests_properties(udonblas-device-execution PROPERTIES TIMEOUT 600)
  add_test(NAME udondnn-device-execution
    COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/boundary/udondnn_device.py"
      "${PROJECT_BINARY_DIR}" "${PROJECT_BINARY_DIR}/tests/udondnn-device")
  set_tests_properties(udondnn-device-execution PROPERTIES TIMEOUT 600)
  set_tests_properties(onnx-runtime-mnist PROPERTIES
    DEPENDS udondnn-device-execution
    ENVIRONMENT
      "UDONCL_ONNX_UNITY_TEMPLATE=${PROJECT_BINARY_DIR}/tests/udondnn-device/unity-udonaten-int4-mm-bundle")
endif()
