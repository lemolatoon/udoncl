foreach(_required IN ITEMS UDONCL_COMPILER UDONCL_CONFIG UDONCL_HOST_MODULE
                           UDONCL_OUTPUT_DIRECTORY UDONCL_BUILD_ROOT
                           UDONCL_RUNTIME_VALIDATION)
  if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
    message(FATAL_ERROR "${_required} is required")
  endif()
endforeach()

if(UDONCL_RUNTIME_VALIDATION)
  set(_runtime_validation_argument --runtime-validation)
else()
  set(_runtime_validation_argument --no-runtime-validation)
endif()
cmake_path(IS_PREFIX UDONCL_BUILD_ROOT "${UDONCL_OUTPUT_DIRECTORY}" NORMALIZE
           _output_is_in_build_tree)
if(NOT _output_is_in_build_tree OR
   UDONCL_OUTPUT_DIRECTORY STREQUAL UDONCL_BUILD_ROOT)
  message(FATAL_ERROR
    "Refusing to replace UdonCL output outside the CMake build tree: "
    "${UDONCL_OUTPUT_DIRECTORY}")
endif()

set(_device_argument)
if(DEFINED UDONCL_DEVICE_MODULE AND NOT "${UDONCL_DEVICE_MODULE}" STREQUAL "")
  set(_device_argument --device-llvm "${UDONCL_DEVICE_MODULE}")
endif()
if(DEFINED UDONCL_HOST_MLIR AND NOT "${UDONCL_HOST_MLIR}" STREQUAL "")
  set(_host_arguments --host-mlir "${UDONCL_HOST_MLIR}"
                      --host-library-llvm "${UDONCL_HOST_MODULE}")
else()
  set(_host_arguments --host-llvm "${UDONCL_HOST_MODULE}")
endif()
file(REMOVE_RECURSE "${UDONCL_OUTPUT_DIRECTORY}")
execute_process(
  COMMAND "${UDONCL_COMPILER}" --emit-unity-library-bundle "${UDONCL_CONFIG}"
          ${_host_arguments} ${_device_argument}
          ${_runtime_validation_argument}
          -o "${UDONCL_OUTPUT_DIRECTORY}"
  RESULT_VARIABLE _result
  OUTPUT_VARIABLE _stdout
  ERROR_VARIABLE _stderr)
if(NOT _result EQUAL 0)
  file(REMOVE_RECURSE "${UDONCL_OUTPUT_DIRECTORY}")
  message(FATAL_ERROR
    "udonclc failed with exit code ${_result}\n${_stdout}${_stderr}")
endif()
foreach(_output IN ITEMS manifest.json host.uasm display.shader)
  if(NOT EXISTS "${UDONCL_OUTPUT_DIRECTORY}/${_output}")
    file(REMOVE_RECURSE "${UDONCL_OUTPUT_DIRECTORY}")
    message(FATAL_ERROR "udonclc succeeded without producing ${_output}")
  endif()
endforeach()
