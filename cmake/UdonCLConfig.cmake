include_guard(GLOBAL)
include(CMakeParseArguments)

get_filename_component(_UDONCL_CONFIG_DIR "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY)
get_filename_component(_UDONCL_INSTALL_PREFIX "${_UDONCL_CONFIG_DIR}/../../.." ABSOLUTE)
get_filename_component(_UDONCL_SOURCE_HINT "${_UDONCL_CONFIG_DIR}/.." ABSOLUTE)
if(EXISTS "${_UDONCL_INSTALL_PREFIX}/include/udoncl/unity_buffer.h")
  set(_UDONCL_INCLUDE_DIR "${_UDONCL_INSTALL_PREFIX}/include")
else()
  set(_UDONCL_INCLUDE_DIR "${_UDONCL_SOURCE_HINT}/include")
endif()

set(UDONCLC_EXECUTABLE "${UDONCLC_EXECUTABLE}" CACHE FILEPATH
    "Path to the udonclc artifact compiler")
set(UDONCL_CLANG_EXECUTABLE "${UDONCL_CLANG_EXECUTABLE}" CACHE FILEPATH
    "Path to the pinned Clang executable")
set(UDONCL_LLVM_LINK_EXECUTABLE "${UDONCL_LLVM_LINK_EXECUTABLE}" CACHE FILEPATH
    "Path to the pinned llvm-link executable")

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
  target_include_directories("${target}" INTERFACE "${_UDONCL_INCLUDE_DIR}")
endfunction()

function(udoncl_add_device_library target)
  _udoncl_add_source_library("${target}" DEVICE ${ARGN})
endfunction()

function(_udoncl_append_target_usage target expected_kind sources_var
         dependencies_var flags_var)
  if(NOT TARGET "${target}")
    message(FATAL_ERROR "Unknown UdonCL source target '${target}'")
  endif()
  get_target_property(_kind "${target}" UDONCL_SOURCE_KIND)
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
      DEPENDS "${_source}" ${dependencies}
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
    DEPENDS ${_objects}
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
    DEPENDS ${modules}
    COMMENT "Linking ${kind} LLVM module for ${artifact}"
    COMMAND_EXPAND_LISTS VERBATIM)
  set(${output_var} "${_module}" PARENT_SCOPE)
endfunction()

function(udoncl_add_artifact target)
  set(_one_value CONFIG OUTPUT_DIRECTORY)
  set(_multi_value HOST_LIBRARIES DEVICE_LIBRARIES)
  cmake_parse_arguments(PARSE_ARGV 1 ARG "" "${_one_value}" "${_multi_value}")
  if(ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "Unknown arguments to udoncl_add_artifact: ${ARG_UNPARSED_ARGUMENTS}")
  endif()
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
  foreach(_library IN LISTS ARG_HOST_LIBRARIES)
    set(_host_sources)
    set(_host_dependencies)
    set(_host_flags)
    _udoncl_append_target_usage("${_library}" HOST _host_sources
                                _host_dependencies _host_flags)
    list(REMOVE_DUPLICATES _host_sources)
    list(REMOVE_DUPLICATES _host_dependencies)
    _udoncl_compile_source_target("${target}" HOST "${_library}"
                                  "${_host_sources}" "${_host_dependencies}"
                                  "${_host_flags}" _host_library_module)
    list(APPEND _host_modules "${_host_library_module}")
  endforeach()
  set(_device_modules)
  foreach(_library IN LISTS ARG_DEVICE_LIBRARIES)
    set(_device_sources)
    set(_device_dependencies)
    set(_device_flags)
    _udoncl_append_target_usage("${_library}" DEVICE _device_sources
                                _device_dependencies _device_flags)
    list(REMOVE_DUPLICATES _device_sources)
    list(REMOVE_DUPLICATES _device_dependencies)
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
    DEPENDS "${_config}" "${_host_module}" "${_device_module}"
    BYPRODUCTS "${_artifact_directory}/manifest.json"
    COMMENT "Generating UdonCL artifact ${target}"
    VERBATIM)
  add_custom_target("${target}" DEPENDS "${_stamp}")
  set_property(TARGET "${target}" PROPERTY
               UDONCL_ARTIFACT_DIRECTORY "${_artifact_directory}")
endfunction()

function(udoncl_add_unity_library target)
  set(_one_value CONFIG OUTPUT_DIRECTORY)
  set(_multi_value HOST_LIBRARIES)
  cmake_parse_arguments(PARSE_ARGV 1 ARG "" "${_one_value}" "${_multi_value}")
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
  foreach(_library IN LISTS ARG_HOST_LIBRARIES)
    set(_sources)
    set(_dependencies)
    set(_flags)
    _udoncl_append_target_usage("${_library}" HOST _sources _dependencies _flags)
    list(REMOVE_DUPLICATES _sources)
    list(REMOVE_DUPLICATES _dependencies)
    _udoncl_compile_source_target("${target}" HOST "${_library}"
                                  "${_sources}" "${_dependencies}" "${_flags}"
                                  _library_module)
    list(APPEND _host_modules "${_library_module}")
  endforeach()
  _udoncl_link_module_set("${target}" HOST "${_host_modules}" _host_module)

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
            "-DUDONCL_OUTPUT_DIRECTORY=${_artifact_directory}"
            "-DUDONCL_BUILD_ROOT=${CMAKE_BINARY_DIR}"
            -P "${_UDONCL_CONFIG_DIR}/UdonCLCompileUnityLibrary.cmake"
    COMMAND "${CMAKE_COMMAND}" -E touch "${_stamp}"
    DEPENDS "${_config}" "${_host_module}"
    BYPRODUCTS "${_artifact_directory}/manifest.json"
               "${_artifact_directory}/host.uasm"
               "${_artifact_directory}/display.shader"
    COMMENT "Generating importable UdonCL Unity library ${target}"
    VERBATIM)
  add_custom_target("${target}" DEPENDS "${_stamp}")
  set_property(TARGET "${target}" PROPERTY
               UDONCL_ARTIFACT_DIRECTORY "${_artifact_directory}")
endfunction()
