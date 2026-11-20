find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/lit/lit.site.cfg.py.in"
  "${CMAKE_CURRENT_BINARY_DIR}/lit/lit.site.cfg.py"
  @ONLY)
add_test(NAME mlir-lit
  COMMAND "${CMAKE_COMMAND}" -E env
    "UDONCL_LIT_VRC_OPT=$<TARGET_FILE:vrc-opt>"
    "UDONCL_LIT_FILECHECK=$<TARGET_FILE:FileCheck>"
    "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/scripts/run_lit.py" -sv
    "${CMAKE_CURRENT_BINARY_DIR}/lit")
add_executable(host-storage-allocation-test
  analysis/host_storage_allocation.cpp)
target_link_libraries(host-storage-allocation-test PRIVATE
  VRCAnalysisHostStorage)
vrc_configure_target(host-storage-allocation-test)
add_test(NAME host-storage-allocation
  COMMAND host-storage-allocation-test)
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
  "${PROJECT_SOURCE_DIR}/libraries/udonblas/src/udonblas.c"
  "${PROJECT_SOURCE_DIR}/libraries/udondnn/src/udondnn.c"
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
  "${PROJECT_SOURCE_DIR}/libraries/udonblas/include"
  "${PROJECT_SOURCE_DIR}/libraries/udondnn/include"
  "${PROJECT_SOURCE_DIR}/libraries/udononnxoperators/include"
  "${PROJECT_SOURCE_DIR}/libraries/udononnxruntime/include"
  "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_compile_definitions(opencl-aot-runtime PRIVATE
  CL_TARGET_OPENCL_VERSION=120
  UDONBLAS_AOT_DIRECT_DISPATCH=1
  UDONBLAS_DEVICE_POINTERS=1
  UDONDNN_AOT_LARGE_ONLY=1
  UDONDNN_DEVICE_POINTERS=1
  UDONCL_AOT_RUNTIME_VALIDATION=1
  UDONCL_AOT_ENABLE_DNN=1
  UDONCL_AOT_ENABLE_MNIST=1
  UDONCL_AOT_ENABLE_ONNX=1)
add_test(NAME opencl-aot-runtime COMMAND opencl-aot-runtime)
add_executable(udontokenizers-runtime
  runtime/udontokenizers.c
  "${PROJECT_SOURCE_DIR}/libraries/udontokenizers/src/udontokenizers.c")
target_compile_features(udontokenizers-runtime PRIVATE c_std_11)
set_target_properties(udontokenizers-runtime PROPERTIES
  C_STANDARD 11 C_STANDARD_REQUIRED YES C_EXTENSIONS OFF)
target_include_directories(udontokenizers-runtime PRIVATE
  "${PROJECT_SOURCE_DIR}/include"
  "${PROJECT_SOURCE_DIR}/libraries/udontokenizers/include")
add_test(NAME udontokenizers-runtime COMMAND udontokenizers-runtime)
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

  add_executable(tinyllama-conversation-input
    runtime/tinyllama_conversation_input.c
    "${PROJECT_SOURCE_DIR}/examples/onnx-transformer-tiny/conversation_input.c"
    "${PROJECT_SOURCE_DIR}/libraries/udontokenizers/src/udontokenizers.c")
  target_compile_features(tinyllama-conversation-input PRIVATE c_std_11)
  set_target_properties(tinyllama-conversation-input PROPERTIES
    C_STANDARD 11 C_STANDARD_REQUIRED YES C_EXTENSIONS OFF)
  target_include_directories(tinyllama-conversation-input PRIVATE
    "${PROJECT_SOURCE_DIR}/third_party/OpenCL-Headers"
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/examples/onnx-transformer-tiny"
    "${PROJECT_SOURCE_DIR}/libraries/udontokenizers/include")
  target_compile_definitions(tinyllama-conversation-input PRIVATE
    CL_TARGET_OPENCL_VERSION=120)
  target_link_libraries(tinyllama-conversation-input PRIVATE VRCOpenCLRuntime)
  add_test(NAME tinyllama-conversation-input
    COMMAND tinyllama-conversation-input)

  add_executable(tinyllama-conversation-host
    runtime/tinyllama_conversation_host.c
    "${PROJECT_SOURCE_DIR}/examples/onnx-transformer-tiny/conversation_host.c")
  target_compile_features(tinyllama-conversation-host PRIVATE c_std_11)
  target_include_directories(tinyllama-conversation-host PRIVATE
    "${PROJECT_SOURCE_DIR}/third_party/OpenCL-Headers"
    "${PROJECT_SOURCE_DIR}/include")
  target_compile_definitions(tinyllama-conversation-host PRIVATE
    CL_TARGET_OPENCL_VERSION=120)
  target_link_libraries(tinyllama-conversation-host PRIVATE VRCOpenCLRuntime)
  add_test(NAME tinyllama-conversation-host COMMAND tinyllama-conversation-host)
endif()
add_executable(udononnx-matmul-nbits-kernel
  boundary/udononnx_matmul_nbits.cpp)
target_compile_features(udononnx-matmul-nbits-kernel PRIVATE cxx_std_20)
set_target_properties(udononnx-matmul-nbits-kernel PROPERTIES
  CXX_EXTENSIONS OFF)
# The test requires bit-identical results between the Q4 kernel and its
# references of the same summation order (the K-tiled serial order for one
# split, the documented split-K order otherwise). The Device lowering never
# contracts (all floats are precise), so the native build must not fuse
# multiply-adds either.
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  target_compile_options(udononnx-matmul-nbits-kernel PRIVATE /fp:precise)
elseif(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
  target_compile_options(udononnx-matmul-nbits-kernel PRIVATE
    /clang:-ffp-contract=off)
else()
  target_compile_options(udononnx-matmul-nbits-kernel PRIVATE
    -ffp-contract=off)
endif()
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
add_executable(udonaten-argmax-kernel
  boundary/udonaten_argmax.cpp)
target_compile_features(udonaten-argmax-kernel PRIVATE cxx_std_20)
set_target_properties(udonaten-argmax-kernel PROPERTIES
  CXX_EXTENSIONS OFF)
add_test(NAME udonaten-argmax-kernel
  COMMAND udonaten-argmax-kernel)
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
add_test(NAME static-device-resource-discovery
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_device_resources.py"
    $<TARGET_FILE:vrc-opt>
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_device_resources.mlir"
)
add_test(NAME static-device-resource-discovery-diagnostics
  COMMAND vrc-opt --vrc-discover-static-device-resources --split-input-file
    --verify-diagnostics
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_device_resources_invalid.mlir"
)
add_test(NAME static-device-resource-context-plan
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_device_resource_context_plan.py"
    $<TARGET_FILE:vrc-opt>
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_device_resource_context_plan.mlir"
)
add_test(NAME static-device-resource-context-plan-diagnostics
  COMMAND vrc-opt
    --vrc-plan-static-device-resource-context-specialization
    --split-input-file --verify-diagnostics
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_device_resource_context_plan_invalid.mlir"
)
add_test(NAME static-allocation-context-specialization
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_allocation_context_specialization.py"
    $<TARGET_FILE:vrc-opt>
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_allocation_context_specialization.mlir"
)
add_test(NAME static-allocation-context-specialization-diagnostics
  COMMAND vrc-opt --vrc-specialize-static-allocation-contexts
    --split-input-file --verify-diagnostics
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_allocation_context_specialization_invalid.mlir"
)
add_test(NAME static-device-resource-requirements
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_device_resource_requirements.py"
    $<TARGET_FILE:vrc-opt>
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_device_resource_requirements.mlir"
)
add_test(NAME static-device-resource-requirements-diagnostics
  COMMAND vrc-opt --vrc-discover-static-device-resources
    --vrc-infer-static-device-resource-requirements --split-input-file
    --verify-diagnostics
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_device_resource_requirements_invalid.mlir"
)
add_test(NAME static-device-resource-provenance
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_device_resource_provenance.py"
    $<TARGET_FILE:vrc-opt>
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_device_resource_provenance.mlir"
)
add_test(NAME static-device-resource-provenance-diagnostics
  COMMAND vrc-opt --vrc-discover-static-device-resources
    --vrc-propagate-device-resource-provenance --split-input-file
    --verify-diagnostics
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_device_resource_provenance_invalid.mlir"
)
add_test(NAME static-device-resource-provenance-missing-discovery
  COMMAND vrc-opt --vrc-propagate-device-resource-provenance
    --split-input-file --verify-diagnostics
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_device_resource_provenance_missing_discovery.mlir"
)
add_test(NAME onnx-public-device-resources
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/onnx_public_device_resources.py"
    $<TARGET_FILE:vrc-opt>
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/onnx_public_device_resources.mlir"
)
add_test(NAME onnx-public-device-resources-diagnostics
  COMMAND vrc-opt --vrc-classify-onnx-public-device-resources
    --split-input-file --verify-diagnostics
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/onnx_public_device_resources_invalid.mlir"
)
add_test(NAME static-device-resource-persistence
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_device_resource_persistence.py"
    $<TARGET_FILE:vrc-opt>
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_device_resource_persistence.mlir"
)
add_test(NAME static-device-resource-persistence-diagnostics
  COMMAND vrc-opt --vrc-analyze-static-device-resource-persistence
    --split-input-file --verify-diagnostics
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_device_resource_persistence_invalid.mlir"
)
add_test(NAME static-device-resource-inventory
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_device_resource_inventory.py"
    $<TARGET_FILE:vrc-opt>
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_device_resource_inventory.mlir"
)
add_test(NAME static-device-resource-inventory-diagnostics
  COMMAND vrc-opt --vrc-build-static-device-resource-inventory
    --split-input-file --verify-diagnostics
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_device_resource_inventory_invalid.mlir"
)
add_test(NAME static-kernel-bindings
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_kernel_bindings.py"
    $<TARGET_FILE:vrc-opt>
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_kernel_bindings.mlir"
)
add_test(NAME static-kernel-bindings-diagnostics
  COMMAND vrc-opt --vrc-discover-static-device-resources
    --vrc-propagate-device-resource-provenance
    --vrc-resolve-static-kernel-bindings --split-input-file
    --verify-diagnostics
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_kernel_bindings_invalid.mlir"
)
add_test(NAME static-dispatch-bindings
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_dispatch_bindings.py"
    $<TARGET_FILE:vrc-opt>
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_dispatch_bindings.mlir"
)
add_test(NAME static-dispatch-bindings-diagnostics
  COMMAND vrc-opt --vrc-resolve-static-kernel-bindings --split-input-file
    --verify-diagnostics
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_dispatch_bindings_invalid.mlir"
)
add_test(NAME static-transfer-bindings
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_transfer_bindings.py"
    $<TARGET_FILE:vrc-opt>
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_transfer_bindings.mlir"
)
add_test(NAME static-transfer-bindings-diagnostics
  COMMAND vrc-opt --vrc-resolve-static-transfer-bindings --split-input-file
    --verify-diagnostics
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_transfer_bindings_invalid.mlir"
)
add_test(NAME static-public-resource-aliases
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_public_resource_aliases.py"
    $<TARGET_FILE:vrc-opt>
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_public_resource_aliases.mlir"
)
add_test(NAME static-public-resource-aliases-diagnostics
  COMMAND vrc-opt --vrc-resolve-static-public-resource-aliases --split-input-file
    --verify-diagnostics
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_public_resource_aliases_invalid.mlir"
)
add_test(NAME static-unity-buffer-reconciliation
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_unity_buffer_reconciliation.py"
    $<TARGET_FILE:vrc-opt>
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_unity_buffer_reconciliation.mlir"
)
add_test(NAME static-unity-buffer-reconciliation-diagnostics
  COMMAND vrc-opt --vrc-reconcile-static-unity-buffers --split-input-file
    --verify-diagnostics
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_unity_buffer_reconciliation_invalid.mlir"
)
add_test(NAME static-unity-buffer-layout-plan
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_unity_buffer_layout_plan.py"
    $<TARGET_FILE:vrc-opt>
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_unity_buffer_layout_plan.mlir"
)
add_test(NAME static-unity-buffer-layout-plan-diagnostics
  COMMAND vrc-opt --vrc-plan-static-unity-buffer-layout --split-input-file
    --verify-diagnostics
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_unity_buffer_layout_plan_invalid.mlir"
)
add_test(NAME unity-texture-assignment
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/unity_texture_assignment.py"
    $<TARGET_FILE:vrc-opt>
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/unity_texture_assignment.mlir"
)
add_test(NAME unity-texture-assignment-diagnostics
  COMMAND vrc-opt --vrc-allocate-unity-textures --split-input-file
    --verify-diagnostics
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/unity_texture_assignment_invalid.mlir"
)
add_test(NAME static-logical-buffer-materialization
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_logical_buffer_materialization.py"
    $<TARGET_FILE:vrc-opt>
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_logical_buffer_materialization.mlir"
)
add_test(NAME host-rematerialization-inventory
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/host_rematerialization_inventory.py"
    $<TARGET_FILE:vrc-opt>
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/host_rematerialization_inventory.mlir"
)
add_test(NAME host-backend-literal-exposure
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/host_backend_literal_exposure.py"
    $<TARGET_FILE:vrc-opt>
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/host_backend_literal_exposure.mlir"
)
add_test(NAME literal-materialization-contract
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/literal_materialization_contract.py"
    $<TARGET_FILE:vrc-opt>
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/literal_materialization_contract.mlir"
)
add_test(NAME host-storage-materialization
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/host_storage_materialization.py"
    $<TARGET_FILE:vrc-opt>
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/host_storage_materialization.mlir"
)
add_test(NAME host-literal-materialization-codegen
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/codegen/host_literal_materialization.py"
    $<TARGET_FILE:vrc-opt>
)
add_test(NAME host-backend-literal-dispatch
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/codegen/host_backend_literal_dispatch.py"
    $<TARGET_FILE:vrc-opt>
)
add_test(NAME static-dispatch-resource-materialization
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_dispatch_resource_materialization.py"
    $<TARGET_FILE:vrc-opt>
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_dispatch_resource_materialization.mlir"
)
add_test(NAME static-dispatch-resource-materialization-diagnostics
  COMMAND vrc-opt --vrc-materialize-static-dispatch-resources
    --split-input-file --verify-diagnostics
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_dispatch_resource_materialization_invalid.mlir"
)
add_test(NAME static-unity-resource-lowering-readiness
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_unity_resource_lowering_readiness.py"
    $<TARGET_FILE:vrc-opt>
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_unity_resource_lowering_readiness.mlir"
)
add_test(NAME static-unity-resource-lowering-readiness-diagnostics
  COMMAND vrc-opt --vrc-assess-static-unity-resource-lowering
    --verify-diagnostics
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_unity_resource_lowering_readiness_invalid.mlir"
)
add_test(NAME static-physical-allocation-specialization-plan
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_physical_allocation_specialization_plan.py"
    $<TARGET_FILE:vrc-opt>
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_physical_allocation_specialization_plan.mlir"
)
add_test(NAME static-physical-allocation-specialization-plan-diagnostics
  COMMAND vrc-opt --vrc-plan-static-physical-allocation-specialization
    --verify-diagnostics
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_physical_allocation_specialization_plan_invalid.mlir"
)
add_test(NAME static-physical-allocation-path-specialization
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_physical_allocation_path_specialization.py"
    $<TARGET_FILE:vrc-opt>
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_physical_allocation_specialization_plan.mlir"
)
add_test(NAME static-physical-allocation-path-specialization-diagnostics
  COMMAND vrc-opt --vrc-specialize-static-physical-allocation-paths
    --verify-diagnostics
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_physical_allocation_path_specialization_invalid.mlir"
)
add_test(NAME static-physical-allocation-materialization
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_physical_allocation_materialization.py"
    $<TARGET_FILE:vrc-opt>
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_physical_allocation_specialization_plan.mlir"
)
add_test(NAME static-physical-allocation-materialization-diagnostics
  COMMAND vrc-opt --vrc-materialize-static-physical-allocations
    --verify-diagnostics
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_physical_allocation_materialization_invalid.mlir"
)
add_test(NAME static-physical-resolve-materialization
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_physical_resolve_materialization.py"
    $<TARGET_FILE:vrc-opt>
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_physical_resolve_materialization.mlir"
)
add_test(NAME checked-assigned-buffer-resolve-lowering
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/checked_assigned_buffer_resolve_lowering.py"
    $<TARGET_FILE:vrc-opt>
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/checked_assigned_buffer_resolve_lowering.mlir"
)
add_test(NAME fixed-physical-release-materialization
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/fixed_physical_release_materialization.py"
    $<TARGET_FILE:vrc-opt>
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/fixed_physical_release_materialization.mlir"
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/fixed_physical_release_materialization_generic.mlir"
)
add_test(NAME dead-physical-allocation-fallback-elimination
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/dead_physical_allocation_fallback_elimination.py"
    $<TARGET_FILE:vrc-opt>
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/dead_physical_allocation_fallback_elimination.mlir"
)
add_test(NAME static-device-resource-switch
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_device_resource_switch.py"
    $<TARGET_FILE:vrc-opt>
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_device_resource_switch.mlir"
)
add_test(NAME static-device-resource-known-control-flow
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_device_resource_known_control_flow.py"
    $<TARGET_FILE:vrc-opt>
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_device_resource_known_control_flow.mlir"
)
add_test(NAME static-device-resource-branch-chain
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_device_resource_branch_chain.py"
    $<TARGET_FILE:vrc-opt>
)
add_test(NAME static-device-resource-local-cells
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_device_resource_local_cells.py"
    $<TARGET_FILE:vrc-opt>
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_device_resource_local_cells.mlir"
)
add_test(NAME static-device-resource-local-cell-diagnostics
  COMMAND vrc-opt --vrc-discover-static-device-resources
    --vrc-propagate-device-resource-provenance --split-input-file
    --verify-diagnostics
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_device_resource_local_cells_invalid.mlir"
)
add_test(NAME static-device-resource-global-cells
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_device_resource_global_cells.py"
    $<TARGET_FILE:vrc-opt>
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_device_resource_global_cells.mlir"
)
add_test(NAME static-device-resource-global-cell-diagnostics
  COMMAND vrc-opt --vrc-discover-static-device-resources
    --vrc-propagate-device-resource-provenance --split-input-file
    --verify-diagnostics
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_device_resource_global_cells_invalid.mlir"
)
add_test(NAME static-device-resource-device-memory
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_device_resource_device_memory.py"
    $<TARGET_FILE:vrc-opt>
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_device_resource_device_memory.mlir"
)
add_test(NAME static-device-resource-device-create
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_device_resource_device_create.py"
    $<TARGET_FILE:vrc-opt>
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_device_resource_device_create.mlir"
)
add_test(NAME static-device-resource-buffer-view
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_device_resource_buffer_view.py"
    $<TARGET_FILE:vrc-opt>
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_device_resource_buffer_view.mlir"
)
add_test(NAME static-device-resource-output-cell
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_device_resource_output_cell.py"
    $<TARGET_FILE:vrc-opt>
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_device_resource_output_cell.mlir"
)
add_test(NAME static-device-resource-aggregate-pointer
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_device_resource_aggregate_pointer.py"
    $<TARGET_FILE:vrc-opt>
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static_device_resource_aggregate_pointer.mlir"
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
add_test(NAME udononnx-matmul-nbits-q4-lowering
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/boundary/udononnx_matmul_nbits_q4_bundle.py"
    $<TARGET_FILE:vrccl> "${VRC_TEST_CLANG}"
    "${PROJECT_BINARY_DIR}/tests/udononnx-matmul-nbits-q4"
)
add_test(NAME unity-synthesized-kernel-bundle
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/boundary/unity_synthesized_dispatch.py"
    $<TARGET_FILE:vrccl> "${VRC_TEST_CLANG}"
    "${PROJECT_BINARY_DIR}/tests/unity-synthesized-kernel"
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

add_test(NAME onnx-oracle-sidecar
  COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/build/onnx_oracle_sidecar.py"
    "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}/tests/onnx-oracle-sidecar"
)
add_test(NAME onnx-cpp-lowering
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/build/onnx_cpp_lowering.py"
    $<TARGET_FILE:vrc-opt>
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/onnx_to_udon_host.mlir"
)
add_test(NAME tinyllama-real-onnx-frontend
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/boundary/tinyllama_onnx_frontend.py"
    "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}/tests/tinyllama-real-onnx-frontend"
    $<TARGET_FILE:udoncl-onnx-frontend>
)
set_tests_properties(tinyllama-real-onnx-frontend PROPERTIES TIMEOUT 600)
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
add_test(NAME tinyllama-tokenizer-cache
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/build/fetch_tinyllama_tokenizer.py"
)
add_test(NAME tinyllama-prefill-artifact-validator
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/build/validate_tinyllama_prefill_artifact.py"
    "${PROJECT_SOURCE_DIR}"
)
add_test(NAME tinyllama-focused-driver
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/build/tinyllama_focused_driver.py"
    "${PROJECT_SOURCE_DIR}"
)
add_test(NAME tinyllama-conversation-artifact-validator
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/build/validate_tinyllama_conversation_artifact.py"
    "${PROJECT_SOURCE_DIR}"
)
add_test(NAME tokenizer-table-compiler
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/build/compile_tokenizer_tables.py"
)
add_test(NAME tinyllama-tokenizer-tables
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/boundary/tinyllama_tokenizer_tables.py"
    "${PROJECT_SOURCE_DIR}"
    "${PROJECT_BINARY_DIR}/tests/tinyllama-tokenizer-tables"
)
add_test(NAME tinyllama-tokenizer-oracle
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/boundary/tinyllama_tokenizer_oracle.py"
    "${PROJECT_SOURCE_DIR}"
    "${PROJECT_BINARY_DIR}/tests/tinyllama-tokenizer-oracle"
    "${CMAKE_C_COMPILER}"
)
add_test(NAME tinyllama-conversation-text
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/boundary/tinyllama_conversation_text.py"
    "${PROJECT_SOURCE_DIR}"
    "${PROJECT_BINARY_DIR}/tests/tinyllama-conversation-text"
    "${CMAKE_C_COMPILER}"
)
add_test(NAME build-parallelism
  COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/build/parallel.py"
)
add_test(NAME repository-line-endings
  COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/build/line_endings.py"
    "${PROJECT_SOURCE_DIR}"
)
add_test(NAME format-script-batches
  COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/build/format_script.py"
)
add_test(NAME commit-validation-service
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/build/precheck_service.py"
    "${PROJECT_SOURCE_DIR}"
)
add_test(NAME precommit-unity-menus
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/build/precommit_unity_menus.py"
    "${PROJECT_SOURCE_DIR}"
)
add_test(NAME commit-validation-cleanup
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/build/precheck_cleanup.py"
    "${PROJECT_SOURCE_DIR}"
)
add_test(NAME onnx-q4-assigned-buffer-component
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/build/onnx_q4_component.py"
    --root "${PROJECT_SOURCE_DIR}"
    --project-build "${PROJECT_BINARY_DIR}"
    --output "${PROJECT_BINARY_DIR}/tests/onnx-q4-component"
    --cmake "${CMAKE_COMMAND}"
    --generator "${CMAKE_GENERATOR}"
    --make-program "${CMAKE_MAKE_PROGRAM}"
    --clang "${VRC_TEST_CLANG}"
    --llvm-link "${VRC_TEST_LLVM_LINK}"
    --onnx-frontend $<TARGET_FILE:udoncl-onnx-frontend>
)
set_tests_properties(onnx-q4-assigned-buffer-component PROPERTIES TIMEOUT 120)
add_test(NAME onnx-rms-assigned-buffer-component
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/build/onnx_rms_component.py"
    --root "${PROJECT_SOURCE_DIR}"
    --project-build "${PROJECT_BINARY_DIR}"
    --output "${PROJECT_BINARY_DIR}/tests/onnx-rms-component"
    --cmake "${CMAKE_COMMAND}"
    --generator "${CMAKE_GENERATOR}"
    --make-program "${CMAKE_MAKE_PROGRAM}"
    --clang "${VRC_TEST_CLANG}"
    --llvm-link "${VRC_TEST_LLVM_LINK}"
    --onnx-frontend $<TARGET_FILE:udoncl-onnx-frontend>
)
set_tests_properties(onnx-rms-assigned-buffer-component PROPERTIES TIMEOUT 120)
add_test(NAME onnx-transpose-assigned-buffer-component
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/build/onnx_transpose_component.py"
    --root "${PROJECT_SOURCE_DIR}"
    --project-build "${PROJECT_BINARY_DIR}"
    --output "${PROJECT_BINARY_DIR}/tests/onnx-transpose-component"
    --cmake "${CMAKE_COMMAND}"
    --generator "${CMAKE_GENERATOR}"
    --make-program "${CMAKE_MAKE_PROGRAM}"
    --clang "${VRC_TEST_CLANG}"
    --llvm-link "${VRC_TEST_LLVM_LINK}"
    --onnx-frontend $<TARGET_FILE:udoncl-onnx-frontend>
)
set_tests_properties(onnx-transpose-assigned-buffer-component PROPERTIES TIMEOUT 120)
add_test(NAME static-initialization-artifact
  COMMAND Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/build/static_initialization.py"
    --root "${PROJECT_SOURCE_DIR}"
    --project-build "${PROJECT_BINARY_DIR}"
    --output "${PROJECT_BINARY_DIR}/tests/static-initialization"
    --cmake "${CMAKE_COMMAND}"
    --generator "${CMAKE_GENERATOR}"
    --make-program "${CMAKE_MAKE_PROGRAM}"
    --clang "${VRC_TEST_CLANG}"
    --llvm-link "${VRC_TEST_LLVM_LINK}"
    --onnx-frontend $<TARGET_FILE:udoncl-onnx-frontend>
)
set_tests_properties(static-initialization-artifact PROPERTIES TIMEOUT 180)
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
    --onnx-frontend $<TARGET_FILE:udoncl-onnx-frontend>
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
  add_executable(synthesized-gather-kernel boundary/synthesized_gather.cpp)
  target_compile_features(synthesized-gather-kernel PRIVATE cxx_std_20)
  set_target_properties(synthesized-gather-kernel PROPERTIES
    CXX_EXTENSIONS OFF)
  target_link_libraries(synthesized-gather-kernel PRIVATE
    VRCNativeOpenCLRuntime)
  add_test(NAME synthesized-gather-execution
    COMMAND Python3::Interpreter
      "${CMAKE_CURRENT_SOURCE_DIR}/boundary/synthesized_gather.py"
      $<TARGET_FILE:vrccl> "${VRC_TEST_CLANG}"
      $<TARGET_FILE:synthesized-gather-kernel>
      "${PROJECT_BINARY_DIR}/tests/synthesized-gather")
  set_tests_properties(synthesized-gather-execution PROPERTIES TIMEOUT 600)
  add_executable(onnx-fusion-kernels boundary/onnx_fusion.cpp)
  target_compile_features(onnx-fusion-kernels PRIVATE cxx_std_20)
  set_target_properties(onnx-fusion-kernels PROPERTIES CXX_EXTENSIONS OFF)
  target_link_libraries(onnx-fusion-kernels PRIVATE VRCNativeOpenCLRuntime)
  add_test(NAME onnx-fusion-execution
    COMMAND Python3::Interpreter
      "${CMAKE_CURRENT_SOURCE_DIR}/boundary/onnx_fusion.py"
      $<TARGET_FILE:vrccl> $<TARGET_FILE:vrc-opt> "${VRC_TEST_CLANG}"
      $<TARGET_FILE:onnx-fusion-kernels>
      "${PROJECT_BINARY_DIR}/tests/onnx-fusion")
  set_tests_properties(onnx-fusion-execution PROPERTIES TIMEOUT 600)
endif()
