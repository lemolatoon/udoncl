include_guard(GLOBAL)
include(CMakeParseArguments)

get_filename_component(_UDONCL_CONFIG_DIR "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY)
get_filename_component(_UDONCL_INSTALL_PREFIX "${_UDONCL_CONFIG_DIR}/../../.." ABSOLUTE)
get_filename_component(_UDONCL_SOURCE_HINT "${_UDONCL_CONFIG_DIR}/.." ABSOLUTE)
if(EXISTS "${_UDONCL_INSTALL_PREFIX}/include/udoncl/unity_buffer.h")
  set(_UDONCL_INCLUDE_DIR "${_UDONCL_INSTALL_PREFIX}/include")
  set(_UDONCL_OPENCL_INCLUDE_DIR "${_UDONCL_INSTALL_PREFIX}/include")
  set(_UDONCL_RUNTIME_DIR
      "${_UDONCL_INSTALL_PREFIX}/share/udoncl/runtime")
  set(_UDONCL_ONNX_COMPILER
      "${_UDONCL_INSTALL_PREFIX}/share/udoncl/onnx/udon_onnx.py")
  set(_UDONCL_ONNX_HOST_COMPILER
      "${_UDONCL_INSTALL_PREFIX}/share/udoncl/onnx/udon_onnx_host.py")
  set(_UDONCL_ONNX_MLIR_COMPILER
      "${_UDONCL_INSTALL_PREFIX}/share/udoncl/onnx/compile_onnx_mlir.py")
  set(_UDONCL_ONNX_MLIR_LOWER
      "${_UDONCL_INSTALL_PREFIX}/share/udoncl/onnx/onnx_mlir_lower.py")
else()
  set(_UDONCL_INCLUDE_DIR "${_UDONCL_SOURCE_HINT}/include")
  set(_UDONCL_OPENCL_INCLUDE_DIR
      "${_UDONCL_SOURCE_HINT}/third_party/OpenCL-Headers")
  set(_UDONCL_RUNTIME_DIR "${_UDONCL_SOURCE_HINT}/lib/Runtime/OpenCL")
  set(_UDONCL_ONNX_COMPILER
      "${_UDONCL_SOURCE_HINT}/runtime/onnx/udon_onnx.py")
  set(_UDONCL_ONNX_HOST_COMPILER
      "${_UDONCL_SOURCE_HINT}/runtime/onnx/udon_onnx_host.py")
  set(_UDONCL_ONNX_MLIR_COMPILER
      "${_UDONCL_SOURCE_HINT}/scripts/compile_onnx_mlir.py")
  set(_UDONCL_ONNX_MLIR_LOWER
      "${_UDONCL_SOURCE_HINT}/runtime/onnx/onnx_mlir_lower.py")
endif()

set(UDONCLC_EXECUTABLE "${UDONCLC_EXECUTABLE}" CACHE FILEPATH
    "Path to the udonclc artifact compiler")
set(UDONCL_CLANG_EXECUTABLE "${UDONCL_CLANG_EXECUTABLE}" CACHE FILEPATH
    "Path to the pinned Clang executable")
set(UDONCL_LLVM_LINK_EXECUTABLE "${UDONCL_LLVM_LINK_EXECUTABLE}" CACHE FILEPATH
    "Path to the pinned llvm-link executable")
set(UDONCL_ONNX_IMPORTER_EXECUTABLE "${UDONCL_ONNX_IMPORTER_EXECUTABLE}"
    CACHE FILEPATH "Path to the locked udoncl-onnx protobuf importer")
set(UDONCL_ONNX_OPT_EXECUTABLE "${UDONCL_ONNX_OPT_EXECUTABLE}"
    CACHE FILEPATH "Path to the locked onnx-mlir-opt executable")

if(NOT UDONCLC_EXECUTABLE)
  find_program(_UDONCLC_DISCOVERED NAMES udonclc
    HINTS
      "${_UDONCL_INSTALL_PREFIX}/bin"
      "${_UDONCL_SOURCE_HINT}/build/udoncl/tools/vrccl"
    NO_DEFAULT_PATH)
  if(_UDONCLC_DISCOVERED)
    set(UDONCLC_EXECUTABLE "${_UDONCLC_DISCOVERED}" CACHE FILEPATH
        "Path to the udonclc artifact compiler" FORCE)
  endif()
endif()
if(NOT UDONCL_CLANG_EXECUTABLE)
  find_program(_UDONCL_CLANG_DISCOVERED NAMES clang
    HINTS
      "${_UDONCL_INSTALL_PREFIX}/bin"
      "${_UDONCL_SOURCE_HINT}/.deps/llvm23/bin"
    NO_DEFAULT_PATH)
  if(_UDONCL_CLANG_DISCOVERED)
    set(UDONCL_CLANG_EXECUTABLE "${_UDONCL_CLANG_DISCOVERED}" CACHE FILEPATH
        "Path to the pinned Clang executable" FORCE)
  endif()
endif()
if(NOT UDONCL_LLVM_LINK_EXECUTABLE)
  find_program(_UDONCL_LLVM_LINK_DISCOVERED NAMES llvm-link
    HINTS
      "${_UDONCL_INSTALL_PREFIX}/bin"
      "${_UDONCL_SOURCE_HINT}/.deps/llvm23/bin"
    NO_DEFAULT_PATH)
  if(_UDONCL_LLVM_LINK_DISCOVERED)
    set(UDONCL_LLVM_LINK_EXECUTABLE "${_UDONCL_LLVM_LINK_DISCOVERED}"
        CACHE FILEPATH "Path to the pinned llvm-link executable" FORCE)
  endif()
endif()
if(NOT UDONCL_ONNX_IMPORTER_EXECUTABLE)
  find_program(_UDONCL_ONNX_IMPORTER_DISCOVERED NAMES udoncl-onnx
    HINTS
      "${_UDONCL_INSTALL_PREFIX}/bin"
      "${_UDONCL_SOURCE_HINT}/.deps/onnx-mlir/bin"
    NO_DEFAULT_PATH)
  if(_UDONCL_ONNX_IMPORTER_DISCOVERED)
    set(UDONCL_ONNX_IMPORTER_EXECUTABLE
        "${_UDONCL_ONNX_IMPORTER_DISCOVERED}" CACHE FILEPATH
        "Path to the locked udoncl-onnx protobuf importer" FORCE)
  endif()
endif()
if(NOT UDONCL_ONNX_OPT_EXECUTABLE)
  find_program(_UDONCL_ONNX_OPT_DISCOVERED NAMES onnx-mlir-opt
    HINTS
      "${_UDONCL_INSTALL_PREFIX}/bin"
      "${_UDONCL_SOURCE_HINT}/.deps/onnx-mlir/bin"
    NO_DEFAULT_PATH)
  if(_UDONCL_ONNX_OPT_DISCOVERED)
    set(UDONCL_ONNX_OPT_EXECUTABLE "${_UDONCL_ONNX_OPT_DISCOVERED}"
        CACHE FILEPATH "Path to the locked onnx-mlir-opt executable" FORCE)
  endif()
endif()

foreach(_tool IN ITEMS UDONCLC_EXECUTABLE UDONCL_CLANG_EXECUTABLE
                       UDONCL_LLVM_LINK_EXECUTABLE)
  if(NOT ${_tool})
    message(FATAL_ERROR
      "${_tool} was not found. Set it to the matching UdonCL compiler tool.")
  endif()
  if(NOT EXISTS "${${_tool}}")
    message(FATAL_ERROR "${_tool} does not exist: ${${_tool}}")
  endif()
endforeach()

function(_udoncl_add_source_library target kind)
  if(TARGET "${target}")
    message(FATAL_ERROR "UdonCL target '${target}' already exists")
  endif()
  add_library("${target}" INTERFACE)
  if(ARGN)
    target_sources("${target}" INTERFACE ${ARGN})
  endif()
  set_property(TARGET "${target}" PROPERTY UDONCL_SOURCE_KIND "${kind}")
endfunction()

function(udoncl_add_host_library target)
  _udoncl_add_source_library("${target}" HOST ${ARGN})
  target_include_directories("${target}" INTERFACE
    "${_UDONCL_INCLUDE_DIR}"
    "${_UDONCL_OPENCL_INCLUDE_DIR}")
endfunction()

function(udoncl_add_device_library target)
  _udoncl_add_source_library("${target}" DEVICE ${ARGN})
endfunction()

function(udoncl_add_onnx_mlir target)
  set(_options)
  set(_one_value MODEL)
  set(_multi_value SUPPORTED_OPERATORS)
  cmake_parse_arguments(PARSE_ARGV 1 _ARG "${_options}" "${_one_value}"
                        "${_multi_value}")
  if(_ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "udoncl_add_onnx_mlir received unknown arguments: ${_ARG_UNPARSED_ARGUMENTS}")
  endif()
  if(NOT _ARG_MODEL)
    message(FATAL_ERROR "udoncl_add_onnx_mlir requires MODEL")
  endif()
  cmake_path(ABSOLUTE_PATH _ARG_MODEL BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
             NORMALIZE OUTPUT_VARIABLE _model)
  if(NOT EXISTS "${_model}")
    message(FATAL_ERROR "ONNX model does not exist: ${_model}")
  endif()
  foreach(_tool IN ITEMS UDONCL_ONNX_IMPORTER_EXECUTABLE
                         UDONCL_ONNX_OPT_EXECUTABLE)
    if(NOT ${_tool} OR NOT EXISTS "${${_tool}}")
      message(FATAL_ERROR
        "${_tool} is required by udoncl_add_onnx_mlir; run scripts/install_onnx_mlir.py or set it explicitly")
    endif()
  endforeach()
  if(NOT EXISTS "${_UDONCL_ONNX_MLIR_COMPILER}")
    message(FATAL_ERROR
      "UdonCL ONNX MLIR pipeline does not exist: ${_UDONCL_ONNX_MLIR_COMPILER}")
  endif()
  find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)
  set(_generated_dir "${CMAKE_CURRENT_BINARY_DIR}/udoncl-onnx/${target}")
  set(_generated_mlir "${_generated_dir}/model.mlir")
  set(_command "${Python3_EXECUTABLE}" "${_UDONCL_ONNX_MLIR_COMPILER}"
      "${_model}" -o "${_generated_mlir}"
      --importer "${UDONCL_ONNX_IMPORTER_EXECUTABLE}"
      --optimizer "${UDONCL_ONNX_OPT_EXECUTABLE}")
  foreach(_operator IN LISTS _ARG_SUPPORTED_OPERATORS)
    list(APPEND _command --supported-operator "${_operator}")
  endforeach()
  add_custom_command(
    OUTPUT "${_generated_mlir}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${_generated_dir}"
    COMMAND ${_command}
    DEPENDS "${_model}" "${_UDONCL_ONNX_MLIR_COMPILER}"
            "${UDONCL_ONNX_IMPORTER_EXECUTABLE}"
            "${UDONCL_ONNX_OPT_EXECUTABLE}"
    VERBATIM
    COMMENT "Canonicalizing ONNX model ${_model}")
  add_custom_target("${target}" DEPENDS "${_generated_mlir}")
  set_property(TARGET "${target}" PROPERTY UDONCL_ONNX_MODEL "${_model}")
  set_property(TARGET "${target}" PROPERTY UDONCL_ONNX_MLIR
               "${_generated_mlir}")
endfunction()

function(udoncl_add_onnx_host_library target)
  set(_options)
  set(_one_value MODEL)
  set(_multi_value DIMENSION_BOUNDS)
  cmake_parse_arguments(PARSE_ARGV 1 _ARG "${_options}" "${_one_value}"
                        "${_multi_value}")
  if(_ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "udoncl_add_onnx_host_library received unknown arguments: ${_ARG_UNPARSED_ARGUMENTS}")
  endif()
  if(NOT _ARG_MODEL)
    message(FATAL_ERROR "udoncl_add_onnx_host_library requires MODEL")
  endif()
  cmake_path(ABSOLUTE_PATH _ARG_MODEL BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
             NORMALIZE OUTPUT_VARIABLE _model)
  if(NOT EXISTS "${_model}")
    message(FATAL_ERROR "ONNX model does not exist: ${_model}")
  endif()
  set(_bound_names)
  foreach(_bound IN LISTS _ARG_DIMENSION_BOUNDS)
    if(NOT _bound MATCHES "^([A-Za-z_][A-Za-z0-9_.]*)=([1-9][0-9]*)$")
      message(FATAL_ERROR
        "Invalid DIMENSION_BOUNDS entry '${_bound}'; expected name=positive_integer")
    endif()
    set(_bound_name "${CMAKE_MATCH_1}")
    if(_bound_name IN_LIST _bound_names)
      message(FATAL_ERROR "Duplicate DIMENSION_BOUNDS entry '${_bound_name}'")
    endif()
    list(APPEND _bound_names "${_bound_name}")
  endforeach()
  foreach(_tool IN ITEMS UDONCL_ONNX_IMPORTER_EXECUTABLE
                         UDONCL_ONNX_OPT_EXECUTABLE)
    if(NOT ${_tool} OR NOT EXISTS "${${_tool}}")
      message(FATAL_ERROR
        "${_tool} is required by udoncl_add_onnx_host_library; run scripts/install_onnx_mlir.py or set it explicitly")
    endif()
  endforeach()
  if(NOT EXISTS "${_UDONCL_ONNX_MLIR_COMPILER}" OR
     NOT EXISTS "${_UDONCL_ONNX_MLIR_LOWER}")
    message(FATAL_ERROR
      "Udon ONNX MLIR Host pipeline is incomplete")
  endif()
  find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)
  set(_generated_dir "${CMAKE_CURRENT_BINARY_DIR}/udoncl-onnx/${target}")
  set(_generated_mlir "${_generated_dir}/model.mlir")
  set(_generated_host "${_generated_dir}/host.c")
  file(MAKE_DIRECTORY "${_generated_dir}")
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${_UDONCL_ONNX_MLIR_COMPILER}"
            "${_model}" -o "${_generated_mlir}"
            --importer "${UDONCL_ONNX_IMPORTER_EXECUTABLE}"
            --optimizer "${UDONCL_ONNX_OPT_EXECUTABLE}"
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr)
  if(NOT _result EQUAL 0)
    message(FATAL_ERROR
      "ONNX MLIR canonicalization failed for ${_model}:\n${_stdout}${_stderr}")
  endif()
  set(_lower_command "${Python3_EXECUTABLE}" "${_UDONCL_ONNX_MLIR_LOWER}"
      "${_model}" "${_generated_mlir}" -o "${_generated_host}")
  foreach(_bound IN LISTS _ARG_DIMENSION_BOUNDS)
    list(APPEND _lower_command --dimension-bound "${_bound}")
  endforeach()
  execute_process(
    COMMAND ${_lower_command}
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr)
  if(NOT _result EQUAL 0)
    message(FATAL_ERROR
      "ONNX Host generation failed for ${_model}:\n${_stdout}${_stderr}")
  endif()
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${_model}" "${_UDONCL_ONNX_MLIR_COMPILER}"
    "${_UDONCL_ONNX_MLIR_LOWER}")
  udoncl_add_host_library("${target}" "${_generated_host}")
  set_property(TARGET "${target}" PROPERTY UDONCL_ONNX_MODEL "${_model}")
  set_property(TARGET "${target}" PROPERTY UDONCL_ONNX_HOST
               "${_generated_host}")
  set_property(TARGET "${target}" PROPERTY UDONCL_ONNX_MLIR
               "${_generated_mlir}")
  set_property(TARGET "${target}" PROPERTY UDONCL_ONNX_DIMENSION_BOUNDS
               "${_ARG_DIMENSION_BOUNDS}")
endfunction()

# Temporary migration-only entry point for the two checked-in MNIST fixtures.
# New consumers must use udoncl_add_onnx_host_library above. Delete this after
# the generic ONNX-MLIR MNIST gate replaces the profile emitter.
function(_udoncl_add_legacy_onnx_host_library target)
  set(_options)
  set(_one_value MODEL PROFILE)
  cmake_parse_arguments(PARSE_ARGV 1 _ARG "${_options}" "${_one_value}" "")
  if(_ARG_UNPARSED_ARGUMENTS OR NOT _ARG_MODEL OR NOT _ARG_PROFILE)
    message(FATAL_ERROR
      "_udoncl_add_legacy_onnx_host_library requires MODEL and PROFILE")
  endif()
  if(NOT _ARG_PROFILE STREQUAL "mnist" AND
     NOT _ARG_PROFILE STREQUAL "mnist-dense")
    message(FATAL_ERROR "Unsupported legacy ONNX Host PROFILE: ${_ARG_PROFILE}")
  endif()
  cmake_path(ABSOLUTE_PATH _ARG_MODEL BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
             NORMALIZE OUTPUT_VARIABLE _model)
  if(NOT EXISTS "${_model}" OR NOT EXISTS "${_UDONCL_ONNX_HOST_COMPILER}")
    message(FATAL_ERROR "Legacy ONNX Host input or compiler does not exist")
  endif()
  find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)
  set(_generated_dir "${CMAKE_CURRENT_BINARY_DIR}/udoncl-onnx/${target}")
  set(_generated_host "${_generated_dir}/host.c")
  set(_generated_mlir "${_generated_dir}/host.mlir")
  file(MAKE_DIRECTORY "${_generated_dir}")
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${_UDONCL_ONNX_HOST_COMPILER}"
            "${_model}" --profile "${_ARG_PROFILE}" -o "${_generated_host}"
            --mlir-output "${_generated_mlir}"
    RESULT_VARIABLE _result OUTPUT_VARIABLE _stdout ERROR_VARIABLE _stderr)
  if(NOT _result EQUAL 0)
    message(FATAL_ERROR
      "Legacy ONNX Host generation failed for ${_model}:\n${_stdout}${_stderr}")
  endif()
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${_model}" "${_UDONCL_ONNX_HOST_COMPILER}")
  udoncl_add_host_library("${target}" "${_generated_host}")
  set_property(TARGET "${target}" PROPERTY UDONCL_ONNX_MODEL "${_model}")
  set_property(TARGET "${target}" PROPERTY UDONCL_ONNX_HOST "${_generated_host}")
  set_property(TARGET "${target}" PROPERTY UDONCL_ONNX_HOST_MLIR
               "${_generated_mlir}")
endfunction()

if(NOT TARGET udoncl_opencl_runtime)
  udoncl_add_host_library(udoncl_opencl_runtime
    "${_UDONCL_RUNTIME_DIR}/OpenCLAOT.c"
    "${_UDONCL_RUNTIME_DIR}/DeviceMemory.c")
  target_compile_definitions(udoncl_opencl_runtime INTERFACE
    CL_TARGET_OPENCL_VERSION=120
    UDONCL_AOT_ENABLE_DNN=0)
endif()
if(NOT TARGET UdonCL::OpenCLRuntime)
  add_library(UdonCL::OpenCLRuntime ALIAS udoncl_opencl_runtime)
endif()
if(NOT TARGET udoncl_opencl_mnist_runtime)
  udoncl_add_host_library(udoncl_opencl_mnist_runtime
    "${_UDONCL_RUNTIME_DIR}/OpenCLAOT.c"
    "${_UDONCL_RUNTIME_DIR}/DeviceMemory.c")
  target_compile_definitions(udoncl_opencl_mnist_runtime INTERFACE
    CL_TARGET_OPENCL_VERSION=120
    UDONCL_AOT_ENABLE_BLAS=0
    UDONCL_AOT_ENABLE_MNIST=1)
endif()
if(NOT TARGET UdonCL::OpenCLMnistRuntime)
  add_library(UdonCL::OpenCLMnistRuntime ALIAS udoncl_opencl_mnist_runtime)
endif()

function(_udoncl_append_target_usage target expected_kind sources_var
         dependencies_var flags_var)
  if(NOT TARGET "${target}")
    message(FATAL_ERROR "Unknown UdonCL source target '${target}'")
  endif()
  get_target_property(_kind "${target}" UDONCL_SOURCE_KIND)
  if(NOT _kind)
    get_target_property(_libraries "${target}"
                        "UDONCL_${expected_kind}_LIBRARIES")
    if(NOT _libraries)
      message(FATAL_ERROR
        "UdonCL target '${target}' does not provide ${expected_kind} sources")
    endif()
    set(_aggregate_sources "${${sources_var}}")
    set(_aggregate_dependencies "${${dependencies_var}}")
    set(_aggregate_flags "${${flags_var}}")
    foreach(_library IN LISTS _libraries)
      _udoncl_append_target_usage("${_library}" "${expected_kind}"
                                  _aggregate_sources _aggregate_dependencies
                                  _aggregate_flags)
    endforeach()
    set(${sources_var} "${_aggregate_sources}" PARENT_SCOPE)
    set(${dependencies_var} "${_aggregate_dependencies}" PARENT_SCOPE)
    set(${flags_var} "${_aggregate_flags}" PARENT_SCOPE)
    return()
  endif()
  if(NOT _kind STREQUAL expected_kind)
    message(FATAL_ERROR
      "UdonCL target '${target}' is ${_kind}; expected ${expected_kind}")
  endif()

  get_target_property(_sources "${target}" INTERFACE_SOURCES)
  if(NOT _sources)
    message(FATAL_ERROR "UdonCL source target '${target}' has no sources")
  endif()
  set(_compiled_sources "${${sources_var}}")
  set(_dependencies "${${dependencies_var}}")
  foreach(_source IN LISTS _sources)
    if(_source MATCHES "\\$<")
      message(FATAL_ERROR
        "Generator expressions in UdonCL source lists are not supported yet: ${_source}")
    endif()
    cmake_path(ABSOLUTE_PATH _source BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
               NORMALIZE OUTPUT_VARIABLE _absolute_source)
    get_filename_component(_extension "${_absolute_source}" LAST_EXT)
    if((expected_kind STREQUAL "HOST" AND _extension STREQUAL ".c") OR
       (expected_kind STREQUAL "DEVICE" AND _extension STREQUAL ".cl"))
      list(APPEND _compiled_sources "${_absolute_source}")
    elseif(_extension MATCHES "^\\.(h|hh|hpp|inc)$")
      list(APPEND _dependencies "${_absolute_source}")
    else()
      message(FATAL_ERROR
        "Unsupported ${expected_kind} source extension in ${_absolute_source}")
    endif()
  endforeach()

  set(_flags "${${flags_var}}")
  get_target_property(_includes "${target}" INTERFACE_INCLUDE_DIRECTORIES)
  if(_includes)
    foreach(_include IN LISTS _includes)
      if(_include MATCHES "\\$<")
        message(FATAL_ERROR
          "Generator expressions in UdonCL include directories are not supported yet: ${_include}")
      endif()
      cmake_path(ABSOLUTE_PATH _include BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                 NORMALIZE OUTPUT_VARIABLE _absolute_include)
      list(APPEND _flags "-I${_absolute_include}")
    endforeach()
  endif()
  get_target_property(_definitions "${target}" INTERFACE_COMPILE_DEFINITIONS)
  if(_definitions)
    foreach(_definition IN LISTS _definitions)
      if(_definition MATCHES "\\$<")
        message(FATAL_ERROR
          "Generator expressions in UdonCL definitions are not supported yet: ${_definition}")
      endif()
      list(APPEND _flags "-D${_definition}")
    endforeach()
  endif()
  get_target_property(_options "${target}" INTERFACE_COMPILE_OPTIONS)
  if(_options)
    foreach(_option IN LISTS _options)
      if(_option MATCHES "\\$<")
        message(FATAL_ERROR
          "Generator expressions in UdonCL compile options are not supported yet: ${_option}")
      endif()
      list(APPEND _flags "${_option}")
    endforeach()
  endif()

  set(${sources_var} "${_compiled_sources}" PARENT_SCOPE)
  set(${dependencies_var} "${_dependencies}" PARENT_SCOPE)
  set(${flags_var} "${_flags}" PARENT_SCOPE)
endfunction()

function(_udoncl_compile_source_target artifact kind unit sources dependencies
         flags output_var)
  set(_objects)
  set(_index 0)
  string(TOLOWER "${kind}" _kind_lower)
  string(MAKE_C_IDENTIFIER "${unit}" _unit_identifier)
  set(_object_directory
      "${CMAKE_CURRENT_BINARY_DIR}/udoncl/${artifact}/objects/${_kind_lower}/${_unit_identifier}")
  foreach(_source IN LISTS sources)
    math(EXPR _index "${_index} + 1")
    get_filename_component(_stem "${_source}" NAME_WE)
    set(_object "${_object_directory}/${_index}-${_stem}.bc")
    if(kind STREQUAL "HOST")
      set(_language_flags -target spir-unknown-unknown -x c -std=c11
          -Xclang -fdefault-calling-conv=cdecl)
    else()
      set(_language_flags -target spir-unknown-unknown -x cl -cl-std=CL1.2
          -cl-kernel-arg-info)
    endif()
    add_custom_command(
      OUTPUT "${_object}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${_object_directory}"
      COMMAND "${UDONCL_CLANG_EXECUTABLE}" ${_language_flags} ${flags}
              -O0 -Xclang -disable-O0-optnone -fomit-frame-pointer
              -ffp-contract=off -emit-llvm -c "${_source}" -o "${_object}"
      DEPENDS "${UDONCL_CLANG_EXECUTABLE}" "${_source}" ${dependencies}
      COMMENT "Compiling ${kind} source ${_source}"
      COMMAND_EXPAND_LISTS VERBATIM)
    list(APPEND _objects "${_object}")
  endforeach()

  if(NOT _objects)
    message(FATAL_ERROR
      "UdonCL ${kind} source target '${unit}' has no compilable sources")
  endif()
  set(_module
      "${CMAKE_CURRENT_BINARY_DIR}/udoncl/${artifact}/modules/${_kind_lower}/${_unit_identifier}.bc")
  add_custom_command(
    OUTPUT "${_module}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory
            "${CMAKE_CURRENT_BINARY_DIR}/udoncl/${artifact}/modules/${_kind_lower}"
    COMMAND "${UDONCL_LLVM_LINK_EXECUTABLE}" ${_objects} -o "${_module}"
    DEPENDS "${UDONCL_LLVM_LINK_EXECUTABLE}" ${_objects}
    COMMENT "Linking ${kind} source target ${unit}"
    COMMAND_EXPAND_LISTS VERBATIM)
  set(${output_var} "${_module}" PARENT_SCOPE)
endfunction()

function(_udoncl_link_module_set artifact kind modules output_var)
  string(TOLOWER "${kind}" _kind_lower)
  set(_module "${CMAKE_CURRENT_BINARY_DIR}/udoncl/${artifact}/${_kind_lower}.bc")
  add_custom_command(
    OUTPUT "${_module}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory
            "${CMAKE_CURRENT_BINARY_DIR}/udoncl/${artifact}"
    COMMAND "${UDONCL_LLVM_LINK_EXECUTABLE}" ${modules} -o "${_module}"
    DEPENDS "${UDONCL_LLVM_LINK_EXECUTABLE}" ${modules}
    COMMENT "Linking ${kind} LLVM module for ${artifact}"
    COMMAND_EXPAND_LISTS VERBATIM)
  set(${output_var} "${_module}" PARENT_SCOPE)
endfunction()

function(udoncl_add_artifact target)
  set(_one_value CONFIG OUTPUT_DIRECTORY)
  set(_multi_value LIBRARIES HOST_LIBRARIES DEVICE_LIBRARIES)
  cmake_parse_arguments(PARSE_ARGV 1 ARG "" "${_one_value}" "${_multi_value}")
  if(ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "Unknown arguments to udoncl_add_artifact: ${ARG_UNPARSED_ARGUMENTS}")
  endif()
  list(APPEND ARG_HOST_LIBRARIES ${ARG_LIBRARIES})
  list(APPEND ARG_DEVICE_LIBRARIES ${ARG_LIBRARIES})
  if(NOT ARG_CONFIG OR NOT ARG_HOST_LIBRARIES OR NOT ARG_DEVICE_LIBRARIES)
    message(FATAL_ERROR
      "udoncl_add_artifact requires CONFIG, HOST_LIBRARIES and DEVICE_LIBRARIES")
  endif()
  if(TARGET "${target}")
    message(FATAL_ERROR "UdonCL artifact target '${target}' already exists")
  endif()

  cmake_path(ABSOLUTE_PATH ARG_CONFIG BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
             NORMALIZE OUTPUT_VARIABLE _config)
  if(NOT EXISTS "${_config}")
    message(FATAL_ERROR "UdonCL artifact config does not exist: ${_config}")
  endif()
  set(_host_modules)
  set(_host_seen_sources)
  foreach(_library IN LISTS ARG_HOST_LIBRARIES)
    set(_host_sources)
    set(_host_dependencies)
    set(_host_flags)
    _udoncl_append_target_usage("${_library}" HOST _host_sources
                                _host_dependencies _host_flags)
    list(REMOVE_DUPLICATES _host_sources)
    list(REMOVE_DUPLICATES _host_dependencies)
    if(_host_seen_sources)
      list(REMOVE_ITEM _host_sources ${_host_seen_sources})
    endif()
    if(NOT _host_sources)
      continue()
    endif()
    list(APPEND _host_seen_sources ${_host_sources})
    _udoncl_compile_source_target("${target}" HOST "${_library}"
                                  "${_host_sources}" "${_host_dependencies}"
                                  "${_host_flags}" _host_library_module)
    list(APPEND _host_modules "${_host_library_module}")
  endforeach()
  set(_device_modules)
  set(_device_seen_sources)
  foreach(_library IN LISTS ARG_DEVICE_LIBRARIES)
    set(_device_sources)
    set(_device_dependencies)
    set(_device_flags)
    _udoncl_append_target_usage("${_library}" DEVICE _device_sources
                                _device_dependencies _device_flags)
    list(REMOVE_DUPLICATES _device_sources)
    list(REMOVE_DUPLICATES _device_dependencies)
    if(_device_seen_sources)
      list(REMOVE_ITEM _device_sources ${_device_seen_sources})
    endif()
    if(NOT _device_sources)
      continue()
    endif()
    list(APPEND _device_seen_sources ${_device_sources})
    _udoncl_compile_source_target("${target}" DEVICE "${_library}"
                                  "${_device_sources}" "${_device_dependencies}"
                                  "${_device_flags}" _device_library_module)
    list(APPEND _device_modules "${_device_library_module}")
  endforeach()
  _udoncl_link_module_set("${target}" HOST "${_host_modules}" _host_module)
  _udoncl_link_module_set("${target}" DEVICE "${_device_modules}"
                          _device_module)

  if(ARG_OUTPUT_DIRECTORY)
    cmake_path(ABSOLUTE_PATH ARG_OUTPUT_DIRECTORY
               BASE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}" NORMALIZE
               OUTPUT_VARIABLE _artifact_directory)
  else()
    set(_artifact_directory
        "${CMAKE_CURRENT_BINARY_DIR}/udoncl/${target}/artifact")
  endif()
  cmake_path(IS_PREFIX CMAKE_BINARY_DIR "${_artifact_directory}" NORMALIZE
             _artifact_is_in_build_tree)
  if(NOT _artifact_is_in_build_tree OR
     _artifact_directory STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR
      "UdonCL OUTPUT_DIRECTORY must be a child of CMAKE_BINARY_DIR because "
      "CMake replaces that directory on rebuild: ${_artifact_directory}")
  endif()
  set(_stamp "${CMAKE_CURRENT_BINARY_DIR}/udoncl/${target}/artifact.stamp")
  add_custom_command(
    OUTPUT "${_stamp}"
    COMMAND "${CMAKE_COMMAND}"
            "-DUDONCL_COMPILER=${UDONCLC_EXECUTABLE}"
            "-DUDONCL_CONFIG=${_config}"
            "-DUDONCL_HOST_MODULE=${_host_module}"
            "-DUDONCL_DEVICE_MODULE=${_device_module}"
            "-DUDONCL_OUTPUT_DIRECTORY=${_artifact_directory}"
            "-DUDONCL_BUILD_ROOT=${CMAKE_BINARY_DIR}"
            -P "${_UDONCL_CONFIG_DIR}/UdonCLCompileArtifact.cmake"
    COMMAND "${CMAKE_COMMAND}" -E touch "${_stamp}"
    DEPENDS "${UDONCLC_EXECUTABLE}" "${_config}" "${_host_module}"
            "${_device_module}"
    BYPRODUCTS "${_artifact_directory}/manifest.json"
    COMMENT "Generating UdonCL artifact ${target}"
    VERBATIM)
  add_custom_target("${target}" DEPENDS "${_stamp}")
  set_property(TARGET "${target}" PROPERTY
               UDONCL_ARTIFACT_DIRECTORY "${_artifact_directory}")
endfunction()

function(udoncl_add_unity_library target)
  set(_one_value CONFIG OUTPUT_DIRECTORY)
  set(_multi_value LIBRARIES HOST_LIBRARIES DEVICE_LIBRARIES)
  cmake_parse_arguments(PARSE_ARGV 1 ARG "" "${_one_value}" "${_multi_value}")
  list(APPEND ARG_HOST_LIBRARIES ${ARG_LIBRARIES})
  list(APPEND ARG_DEVICE_LIBRARIES ${ARG_LIBRARIES})
  if(ARG_UNPARSED_ARGUMENTS OR NOT ARG_CONFIG OR NOT ARG_HOST_LIBRARIES)
    message(FATAL_ERROR
      "udoncl_add_unity_library requires CONFIG and HOST_LIBRARIES")
  endif()
  if(TARGET "${target}")
    message(FATAL_ERROR "UdonCL Unity library target '${target}' already exists")
  endif()
  cmake_path(ABSOLUTE_PATH ARG_CONFIG BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
             NORMALIZE OUTPUT_VARIABLE _config)
  if(NOT EXISTS "${_config}")
    message(FATAL_ERROR "UdonCL Unity library config does not exist: ${_config}")
  endif()

  set(_host_modules)
  set(_host_seen_sources)
  set(_host_mlir)
  foreach(_library IN LISTS ARG_HOST_LIBRARIES)
    get_target_property(_library_host_mlir "${_library}"
                        UDONCL_ONNX_HOST_MLIR)
    if(_library_host_mlir)
      if(_host_mlir)
        message(FATAL_ERROR
          "UdonCL Unity libraries currently support one Host MLIR module")
      endif()
      set(_host_mlir "${_library_host_mlir}")
      continue()
    endif()
    set(_sources)
    set(_dependencies)
    set(_flags)
    _udoncl_append_target_usage("${_library}" HOST _sources _dependencies _flags)
    list(REMOVE_DUPLICATES _sources)
    list(REMOVE_DUPLICATES _dependencies)
    if(_host_seen_sources)
      list(REMOVE_ITEM _sources ${_host_seen_sources})
    endif()
    if(NOT _sources)
      continue()
    endif()
    list(APPEND _host_seen_sources ${_sources})
    _udoncl_compile_source_target("${target}" HOST "${_library}"
                                  "${_sources}" "${_dependencies}" "${_flags}"
                                  _library_module)
    list(APPEND _host_modules "${_library_module}")
  endforeach()
  _udoncl_link_module_set("${target}" HOST "${_host_modules}" _host_module)

  set(_device_module)
  if(ARG_DEVICE_LIBRARIES)
    set(_device_modules)
    set(_device_seen_sources)
    foreach(_library IN LISTS ARG_DEVICE_LIBRARIES)
      set(_sources)
      set(_dependencies)
      set(_flags)
      _udoncl_append_target_usage("${_library}" DEVICE _sources _dependencies
                                  _flags)
      list(REMOVE_DUPLICATES _sources)
      list(REMOVE_DUPLICATES _dependencies)
      if(_device_seen_sources)
        list(REMOVE_ITEM _sources ${_device_seen_sources})
      endif()
      if(NOT _sources)
        continue()
      endif()
      list(APPEND _device_seen_sources ${_sources})
      _udoncl_compile_source_target("${target}" DEVICE "${_library}"
                                    "${_sources}" "${_dependencies}" "${_flags}"
                                    _library_module)
      list(APPEND _device_modules "${_library_module}")
    endforeach()
    _udoncl_link_module_set("${target}" DEVICE "${_device_modules}"
                            _device_module)
  endif()

  if(ARG_OUTPUT_DIRECTORY)
    cmake_path(ABSOLUTE_PATH ARG_OUTPUT_DIRECTORY
               BASE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}" NORMALIZE
               OUTPUT_VARIABLE _artifact_directory)
  else()
    set(_artifact_directory
        "${CMAKE_CURRENT_BINARY_DIR}/udoncl/${target}/artifact")
  endif()
  cmake_path(IS_PREFIX CMAKE_BINARY_DIR "${_artifact_directory}" NORMALIZE
             _artifact_is_in_build_tree)
  if(NOT _artifact_is_in_build_tree OR
     _artifact_directory STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR
      "UdonCL OUTPUT_DIRECTORY must be a child of CMAKE_BINARY_DIR because "
      "CMake replaces that directory on rebuild: ${_artifact_directory}")
  endif()
  set(_stamp "${CMAKE_CURRENT_BINARY_DIR}/udoncl/${target}/artifact.stamp")
  add_custom_command(
    OUTPUT "${_stamp}"
    COMMAND "${CMAKE_COMMAND}"
            "-DUDONCL_COMPILER=${UDONCLC_EXECUTABLE}"
            "-DUDONCL_CONFIG=${_config}"
            "-DUDONCL_HOST_MODULE=${_host_module}"
            "-DUDONCL_HOST_MLIR=${_host_mlir}"
            "-DUDONCL_DEVICE_MODULE=${_device_module}"
            "-DUDONCL_OUTPUT_DIRECTORY=${_artifact_directory}"
            "-DUDONCL_BUILD_ROOT=${CMAKE_BINARY_DIR}"
            -P "${_UDONCL_CONFIG_DIR}/UdonCLCompileUnityLibrary.cmake"
    COMMAND "${CMAKE_COMMAND}" -E touch "${_stamp}"
    DEPENDS "${UDONCLC_EXECUTABLE}" "${_config}" "${_host_module}"
            ${_host_mlir}
            ${_device_module}
    BYPRODUCTS "${_artifact_directory}/manifest.json"
               "${_artifact_directory}/host.uasm"
               "${_artifact_directory}/display.shader"
    COMMENT "Generating importable UdonCL Unity library ${target}"
    VERBATIM)
  add_custom_target("${target}" DEPENDS "${_stamp}")
  set_property(TARGET "${target}" PROPERTY
               UDONCL_ARTIFACT_DIRECTORY "${_artifact_directory}")
endfunction()
