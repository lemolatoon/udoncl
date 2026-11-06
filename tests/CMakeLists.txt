find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)
file(READ
  "${PROJECT_SOURCE_DIR}/libraries/udononnxoperators/device/matmul_nbits.cl"
  _UDONONNX_TEST_DEVICE_SOURCE)
file(READ
  "${PROJECT_SOURCE_DIR}/libraries/udononnxoperators/device/gather.cl"
  _UDONONNX_TEST_GATHER_SOURCE)
file(READ
  "${PROJECT_SOURCE_DIR}/libraries/udononnxoperators/device/unary.cl"
  _UDONONNX_TEST_UNARY_SOURCE)
file(READ
  "${PROJECT_SOURCE_DIR}/libraries/udononnxoperators/device/reindex.cl"
  _UDONONNX_TEST_REINDEX_SOURCE)
string(APPEND _UDONONNX_TEST_DEVICE_SOURCE
  "\n${_UDONONNX_TEST_GATHER_SOURCE}\n${_UDONONNX_TEST_UNARY_SOURCE}\n${_UDONONNX_TEST_REINDEX_SOURCE}")
string(REPLACE "\\" "\\\\" _UDONONNX_TEST_DEVICE_SOURCE
  "${_UDONONNX_TEST_DEVICE_SOURCE}")
string(REPLACE "\"" "\\\"" _UDONONNX_TEST_DEVICE_SOURCE
  "${_UDONONNX_TEST_DEVICE_SOURCE}")
string(REPLACE "\n" "\\n" _UDONONNX_TEST_DEVICE_SOURCE
  "${_UDONONNX_TEST_DEVICE_SOURCE}")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(WRITE
  "${CMAKE_CURRENT_BINARY_DIR}/generated/udononnxoperators_device_source.h"
  "#define UDONONNX_DEVICE_SOURCE \"${_UDONONNX_TEST_DEVICE_SOURCE}\"\n")
set(_UDONBLAS_TEST_DEVICE_SOURCE "")
foreach(_source sgemm.cl sgemv.cl saxpy.cl sdot.cl)
  file(READ
    "${PROJECT_SOURCE_DIR}/libraries/udonblas/device/${_source}"
    _UDONBLAS_TEST_DEVICE_PART)
  string(APPEND _UDONBLAS_TEST_DEVICE_SOURCE
    "${_UDONBLAS_TEST_DEVICE_PART}\n")
endforeach()
string(REPLACE "\\" "\\\\" _UDONBLAS_TEST_DEVICE_SOURCE
  "${_UDONBLAS_TEST_DEVICE_SOURCE}")
string(REPLACE "\"" "\\\"" _UDONBLAS_TEST_DEVICE_SOURCE
  "${_UDONBLAS_TEST_DEVICE_SOURCE}")
string(REPLACE "\n" "\\n" _UDONBLAS_TEST_DEVICE_SOURCE
  "${_UDONBLAS_TEST_DEVICE_SOURCE}")
file(WRITE
  "${CMAKE_CURRENT_BINARY_DIR}/generated/udonblas_device_source.h"
  "#define UDONBLAS_DEVICE_SOURCE \"${_UDONBLAS_TEST_DEVICE_SOURCE}\"\n")
file(WRITE
  "${CMAKE_CURRENT_BINARY_DIR}/generated/udondnn_device_source.h"
  "#define UDONDNN_DEVICE_SOURCE \"udondnn-stream-runtime\"\n")
add_executable(opencl-aot-runtime
  runtime/opencl_aot.c
  "${PROJECT_SOURCE_DIR}/lib/Runtime/OpenCL/OpenCLAOT.c"
  "${PROJECT_SOURCE_DIR}/lib/Runtime/OpenCL/DeviceMemory.c"
  "${PROJECT_SOURCE_DIR}/libraries/udononnxruntime/src/udononnxruntime.c"
  "${PROJECT_SOURCE_DIR}/libraries/udononnxruntime/src/onnx_binding_abi.c"
  "${PROJECT_SOURCE_DIR}/libraries/udononnxoperators/src/matmul_nbits.c")
target_compile_features(opencl-aot-runtime PRIVATE c_std_11)
set_target_properties(opencl-aot-runtime PROPERTIES
  C_STANDARD 11 C_STANDARD_REQUIRED YES C_EXTENSIONS OFF)
target_include_directories(opencl-aot-runtime PRIVATE
  "${PROJECT_SOURCE_DIR}/third_party/OpenCL-Headers"
  "${PROJECT_SOURCE_DIR}/include"
  "${PROJECT_SOURCE_DIR}/libraries/udononnxoperators/include"
  "${PROJECT_SOURCE_DIR}/libraries/udononnxruntime/include"
  "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_compile_definitions(opencl-aot-runtime PRIVATE
  CL_TARGET_OPENCL_VERSION=120
  UDONCL_AOT_RUNTIME_VALIDATION=1
  UDONCL_AOT_ENABLE_MNIST=1
  UDONCL_AOT_ENABLE_ONNX=1)
add_test(NAME opencl-aot-runtime COMMAND opencl-aot-runtime)
if(TARGET VRCOpenCLRuntime)
  add_executable(udonblas-stream-runtime
    runtime/udonblas_stream.c
    "${PROJECT_SOURCE_DIR}/libraries/udonblas/src/udonblas.c")
  target_compile_features(udonblas-stream-runtime PRIVATE c_std_11)
  set_target_properties(udonblas-stream-runtime PROPERTIES
    C_STANDARD 11 C_STANDARD_REQUIRED YES C_EXTENSIONS OFF)
  target_include_directories(udonblas-stream-runtime PRIVATE
    "${PROJECT_SOURCE_DIR}/third_party/OpenCL-Headers"
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/libraries/udonblas/include"
    "${CMAKE_CURRENT_BINARY_DIR}/generated")
  target_compile_definitions(udonblas-stream-runtime PRIVATE
    CL_TARGET_OPENCL_VERSION=120)
  target_link_libraries(udonblas-stream-runtime PRIVATE VRCOpenCLRuntime)
  add_test(NAME udonblas-stream-runtime COMMAND udonblas-stream-runtime)

  add_executable(udondnn-stream-runtime
    runtime/udondnn_stream.c
    "${PROJECT_SOURCE_DIR}/libraries/udondnn/src/udondnn.c")
  target_compile_features(udondnn-stream-runtime PRIVATE c_std_11)
  set_target_properties(udondnn-stream-runtime PROPERTIES
    C_STANDARD 11 C_STANDARD_REQUIRED YES C_EXTENSIONS OFF)
  target_include_directories(udondnn-stream-runtime PRIVATE
    "${PROJECT_SOURCE_DIR}/third_party/OpenCL-Headers"
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/libraries/udondnn/include"
    "${CMAKE_CURRENT_BINARY_DIR}/generated")
  target_compile_definitions(udondnn-stream-runtime PRIVATE
    CL_TARGET_OPENCL_VERSION=120 UDONDNN_AOT_LARGE_ONLY=1)
  target_link_libraries(udondnn-stream-runtime PRIVATE VRCOpenCLRuntime)
  add_test(NAME udondnn-stream-runtime COMMAND udondnn-stream-runtime)

  add_executable(udononnx-runtime
    runtime/onnx_runtime.c
    "${PROJECT_SOURCE_DIR}/libraries/udononnxruntime/src/udononnxruntime.c"
    "${PROJECT_SOURCE_DIR}/libraries/udononnxruntime/src/onnx_binding_abi.c")
  target_compile_features(udononnx-runtime PRIVATE c_std_11)
  set_target_properties(udononnx-runtime PROPERTIES
    C_STANDARD 11 C_STANDARD_REQUIRED YES C_EXTENSIONS OFF)
  target_include_directories(udononnx-runtime PRIVATE
    "${PROJECT_SOURCE_DIR}/third_party/OpenCL-Headers"
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/libraries/udononnxruntime/include")
  target_compile_definitions(udononnx-runtime PRIVATE
    CL_TARGET_OPENCL_VERSION=120)
  target_link_libraries(udononnx-runtime PRIVATE VRCOpenCLRuntime)
  add_test(NAME udononnx-runtime COMMAND udononnx-runtime)
endif()
add_executable(udononnx-matmul-nbits-kernel
  boundary/udononnx_matmul_nbits.cpp)
target_compile_features(udononnx-matmul-nbits-kernel PRIVATE cxx_std_20)
set_target_properties(udononnx-matmul-nbits-kernel PROPERTIES
  CXX_EXTENSIONS OFF)
add_test(NAME udononnx-matmul-nbits-kernel
  COMMAND udononnx-matmul-nbits-kernel)
add_executable(udonblas-arena-sgemm-kernel
  boundary/udonblas_arena_sgemm.cpp)
target_compile_features(udonblas-arena-sgemm-kernel PRIVATE cxx_std_20)
set_target_properties(udonblas-arena-sgemm-kernel PROPERTIES
  CXX_EXTENSIONS OFF)
add_test(NAME udonblas-arena-sgemm-kernel
  COMMAND udonblas-arena-sgemm-kernel)
add_executable(udononnx-reindex-kernel
  boundary/udononnx_reindex.cpp)
target_compile_features(udononnx-reindex-kernel PRIVATE cxx_std_20)
set_target_properties(udononnx-reindex-kernel PROPERTIES
  CXX_EXTENSIONS OFF)
add_test(NAME udononnx-reindex-kernel
  COMMAND udononnx-reindex-kernel)
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
vrc_find_test_tool(VRC_TEST_LLI lli)
vrc_find_test_tool(VRC_TEST_MLIR_OPT mlir-opt)
vrc_find_test_tool(VRC_TEST_MLIR_TRANSLATE mlir-translate)
find_file(VRC_TEST_ASYNC_RUNTIME
  NAMES mlir_async_runtime.dll libmlir_async_runtime.so
        libmlir_async_runtime.dylib
  HINTS "${LLVM_TOOLS_BINARY_DIR}" ${LLVM_LIBRARY_DIRS}
  NO_DEFAULT_PATH REQUIRED)
add_test(NAME opencl-frontend-metadata
  COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/frontend/opencl.py"
    $<TARGET_FILE:vrc-translate> $<TARGET_FILE:vrc-opt>
    "${VRC_TEST_CLANG}" "${VRC_TEST_LLVM_OPT}"
    "${PROJECT_BINARY_DIR}/tests/opencl"
)
add_test(NAME host-to-async-lowering
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/host_to_async.py"
    $<TARGET_FILE:vrc-opt>
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/host_to_async.mlir"
)
if(WIN32)
  add_library(native-async-support SHARED frontend/native_async_support.c)
  set_target_properties(native-async-support PROPERTIES
    C_STANDARD 11 C_STANDARD_REQUIRED YES C_EXTENSIONS OFF)
  set(VRC_TEST_NATIVE_ASYNC_SUPPORT $<TARGET_FILE:native-async-support>)
else()
  set(VRC_TEST_NATIVE_ASYNC_SUPPORT "")
endif()
add_test(NAME native-async-coroutine-execution
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/native_async.py"
    $<TARGET_FILE:vrc-opt>
    "${VRC_TEST_MLIR_OPT}" "${VRC_TEST_MLIR_TRANSLATE}"
    "${VRC_TEST_LLVM_OPT}" "${VRC_TEST_LLI}"
    "${VRC_TEST_ASYNC_RUNTIME}"
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/native_async.mlir"
    "${VRC_TEST_NATIVE_ASYNC_SUPPORT}"
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
add_test(NAME tinyllama-real-onnx-oracle
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/boundary/tinyllama_real_onnx.py"
    "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}/tests/tinyllama-real-onnx"
)
set_tests_properties(tinyllama-real-onnx-oracle PROPERTIES TIMEOUT 600)
if(VRC_TEST_ONNX_IMPORTER AND VRC_TEST_ONNX_OPT)
  add_test(NAME onnx-cpp-lowering
    COMMAND Python3::Interpreter
      "${CMAKE_CURRENT_SOURCE_DIR}/build/onnx_cpp_lowering.py"
      "${VRC_TEST_ONNX_OPT}"
      $<TARGET_FILE:vrc-opt>
      "${CMAKE_CURRENT_SOURCE_DIR}/frontend/onnx_to_udon_host.mlir"
  )
  add_test(NAME tinyllama-real-onnx-frontend
    COMMAND Python3::Interpreter
      "${CMAKE_CURRENT_SOURCE_DIR}/boundary/tinyllama_onnx_frontend.py"
      "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}/tests/tinyllama-real-onnx-frontend"
      "${VRC_TEST_ONNX_IMPORTER}" "${VRC_TEST_ONNX_OPT}"
  )
  set_tests_properties(tinyllama-real-onnx-frontend PROPERTIES TIMEOUT 600)
endif()
add_test(NAME onnx-runtime-mnist
  COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/boundary/onnx_runtime.py"
    "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}/tests/onnx-runtime-mnist"
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
add_test(NAME tinyllama-model-cache
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/build/fetch_tinyllama_onnx.py"
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
    --onnx-importer "${VRC_TEST_ONNX_IMPORTER}"
    --onnx-opt "${VRC_TEST_ONNX_OPT}"
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
