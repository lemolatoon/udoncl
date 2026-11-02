foreach(_required IN ITEMS UDONCL_COMPILER UDONCL_CONFIG UDONCL_HOST_MODULE
                           UDONCL_DEVICE_MODULE UDONCL_OUTPUT_DIRECTORY
                           UDONCL_BUILD_ROOT)
  if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
    message(FATAL_ERROR "${_required} is required")
  endif()
endforeach()

cmake_path(IS_PREFIX UDONCL_BUILD_ROOT "${UDONCL_OUTPUT_DIRECTORY}" NORMALIZE
           _output_is_in_build_tree)
if(NOT _output_is_in_build_tree OR
   UDONCL_OUTPUT_DIRECTORY STREQUAL UDONCL_BUILD_ROOT)
  message(FATAL_ERROR
    "Refusing to replace UdonCL output outside the CMake build tree: "
    "${UDONCL_OUTPUT_DIRECTORY}")
endif()

# The artifact compiler intentionally never mutates an existing artifact.
# CMake owns this exact build-tree directory, so replace it before rebuilding.
file(REMOVE_RECURSE "${UDONCL_OUTPUT_DIRECTORY}")
execute_process(
  COMMAND "${UDONCL_COMPILER}" --emit-llvm-boundary-bundle "${UDONCL_CONFIG}"
          --host-llvm "${UDONCL_HOST_MODULE}"
          --device-llvm "${UDONCL_DEVICE_MODULE}"
          -o "${UDONCL_OUTPUT_DIRECTORY}"
  RESULT_VARIABLE _result
  OUTPUT_VARIABLE _stdout
  ERROR_VARIABLE _stderr)
if(NOT _result EQUAL 0)
  file(REMOVE_RECURSE "${UDONCL_OUTPUT_DIRECTORY}")
  message(FATAL_ERROR
    "udonclc failed with exit code ${_result}\n${_stdout}${_stderr}")
endif()
if(NOT EXISTS "${UDONCL_OUTPUT_DIRECTORY}/manifest.json")
  file(REMOVE_RECURSE "${UDONCL_OUTPUT_DIRECTORY}")
  message(FATAL_ERROR "udonclc succeeded without producing manifest.json")
endif()
