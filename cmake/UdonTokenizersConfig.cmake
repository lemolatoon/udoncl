include_guard(GLOBAL)
include(CMakeFindDependencyMacro)

find_dependency(UdonCL 0.1 CONFIG)

get_filename_component(_UDONTOKENIZERS_CONFIG_DIR "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY)
get_filename_component(_UDONTOKENIZERS_INSTALL_PREFIX "${_UDONTOKENIZERS_CONFIG_DIR}/../../.." ABSOLUTE)
get_filename_component(_UDONTOKENIZERS_SOURCE_HINT "${_UDONTOKENIZERS_CONFIG_DIR}/.." ABSOLUTE)

if(EXISTS "${_UDONTOKENIZERS_INSTALL_PREFIX}/share/udontokenizers/src/udontokenizers.c")
  set(_UDONTOKENIZERS_ROOT "${_UDONTOKENIZERS_INSTALL_PREFIX}/share/udontokenizers")
  set(_UDONTOKENIZERS_INCLUDE_DIR "${_UDONTOKENIZERS_INSTALL_PREFIX}/include")
  set(_UDONTOKENIZERS_TABLE_COMPILER
      "${_UDONTOKENIZERS_ROOT}/compile_tokenizer_tables.py")
else()
  set(_UDONTOKENIZERS_ROOT "${_UDONTOKENIZERS_SOURCE_HINT}/libraries/udontokenizers")
  set(_UDONTOKENIZERS_INCLUDE_DIR "${_UDONTOKENIZERS_ROOT}/include")
  set(_UDONTOKENIZERS_TABLE_COMPILER
      "${_UDONTOKENIZERS_SOURCE_HINT}/scripts/compile_tokenizer_tables.py")
endif()

function(udontokenizers_add_huggingface_model target)
  set(_options)
  set(_one_value TOKENIZER_DIRECTORY SYMBOL)
  set(_multi_value SOURCES)
  cmake_parse_arguments(PARSE_ARGV 1 _ARG "${_options}" "${_one_value}"
                        "${_multi_value}")
  if(_ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "udontokenizers_add_huggingface_model received unknown arguments: ${_ARG_UNPARSED_ARGUMENTS}")
  endif()
  if(NOT _ARG_TOKENIZER_DIRECTORY)
    message(FATAL_ERROR
      "udontokenizers_add_huggingface_model requires TOKENIZER_DIRECTORY")
  endif()
  if(NOT _ARG_SYMBOL)
    message(FATAL_ERROR
      "udontokenizers_add_huggingface_model requires SYMBOL")
  endif()
  string(MAKE_C_IDENTIFIER "${_ARG_SYMBOL}" _symbol_identifier)
  if(NOT _symbol_identifier STREQUAL _ARG_SYMBOL)
    message(FATAL_ERROR
      "udontokenizers_add_huggingface_model SYMBOL must be a C identifier")
  endif()
  cmake_path(ABSOLUTE_PATH _ARG_TOKENIZER_DIRECTORY
             BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" NORMALIZE
             OUTPUT_VARIABLE _tokenizer_directory)
  set(_assets
      "${_tokenizer_directory}/tokenizer.json"
      "${_tokenizer_directory}/tokenizer.model"
      "${_tokenizer_directory}/tokenizer_config.json"
      "${_tokenizer_directory}/special_tokens_map.json")
  foreach(_asset IN LISTS _assets)
    if(NOT EXISTS "${_asset}")
      message(FATAL_ERROR "Tokenizer asset does not exist: ${_asset}")
    endif()
  endforeach()
  if(NOT EXISTS "${_UDONTOKENIZERS_TABLE_COMPILER}")
    message(FATAL_ERROR
      "UdonTokenizers table compiler does not exist: ${_UDONTOKENIZERS_TABLE_COMPILER}")
  endif()
  find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)
  set(_generated_directory
      "${CMAKE_CURRENT_BINARY_DIR}/udontokenizers/${target}")
  set(_generated_source "${_generated_directory}/${_ARG_SYMBOL}.c")
  set(_generated_header "${_generated_directory}/${_ARG_SYMBOL}.h")
  set(_generated_manifest "${_generated_directory}/manifest.json")
  add_custom_command(
    OUTPUT "${_generated_source}" "${_generated_header}"
           "${_generated_manifest}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${_generated_directory}"
    COMMAND "${Python3_EXECUTABLE}" "${_UDONTOKENIZERS_TABLE_COMPILER}"
            "${_tokenizer_directory}" --output "${_generated_directory}"
            --symbol "${_ARG_SYMBOL}"
    DEPENDS "${_UDONTOKENIZERS_TABLE_COMPILER}" ${_assets}
    COMMENT "Compiling Hugging Face tokenizer tables for ${target}"
    VERBATIM)
  set_source_files_properties("${_generated_source}" "${_generated_header}"
                              PROPERTIES GENERATED TRUE)
  udoncl_add_host_library("${target}" "${_generated_source}"
                          "${_generated_header}" ${_ARG_SOURCES})
  target_include_directories("${target}" INTERFACE
    "${_generated_directory}" "${_UDONTOKENIZERS_INCLUDE_DIR}")
  set_property(TARGET "${target}" PROPERTY
    UDONTOKENIZERS_GENERATED_DIRECTORY "${_generated_directory}")
  set_property(TARGET "${target}" PROPERTY
    UDONTOKENIZERS_GENERATED_MANIFEST "${_generated_manifest}")
endfunction()

if(NOT TARGET udontokenizers_host)
  udoncl_add_host_library(udontokenizers_host
    "${_UDONTOKENIZERS_ROOT}/src/udontokenizers.c")
  target_include_directories(udontokenizers_host INTERFACE
    "${_UDONTOKENIZERS_INCLUDE_DIR}")
endif()
if(NOT TARGET udontokenizers)
  add_library(udontokenizers INTERFACE)
  set_property(TARGET udontokenizers PROPERTY UDONCL_HOST_LIBRARIES
    udontokenizers_host)
endif()
if(NOT TARGET UdonTokenizers::UdonTokenizers)
  add_library(UdonTokenizers::UdonTokenizers ALIAS udontokenizers)
endif()

set(UdonTokenizers_VERSION 0.1.0)
set(UdonTokenizers_INCLUDE_DIRS "${_UDONTOKENIZERS_INCLUDE_DIR}")
set(UdonTokenizers_FOUND TRUE)
