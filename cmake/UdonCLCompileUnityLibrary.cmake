foreach(_required IN ITEMS UDONCL_COMPILER
                           UDONCL_OUTPUT_DIRECTORY UDONCL_BUILD_ROOT
                           UDONCL_RUNTIME_VALIDATION)
  if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
    message(FATAL_ERROR "${_required} is required")
  endif()
endforeach()
if((NOT DEFINED UDONCL_HOST_MODULE OR "${UDONCL_HOST_MODULE}" STREQUAL "") AND
   (NOT DEFINED UDONCL_HOST_MLIR OR "${UDONCL_HOST_MLIR}" STREQUAL ""))
  message(FATAL_ERROR "UDONCL_HOST_MODULE or UDONCL_HOST_MLIR is required")
endif()

if(UDONCL_RUNTIME_VALIDATION)
  set(_runtime_validation_argument --runtime-validation)
else()
  set(_runtime_validation_argument --no-runtime-validation)
endif()
set(_intermediate_ir_argument)
if(UDONCL_OMIT_INTERMEDIATE_IR)
  set(_intermediate_ir_argument --omit-intermediate-ir)
endif()
set(_library_arguments)
foreach(_setting IN ITEMS BINDING_NAMESPACE BINDING_CLASS DEVICE_TEXEL_FORMAT
                          DEVICE_WORK_BUDGET)
  if(DEFINED UDONCL_${_setting} AND NOT "${UDONCL_${_setting}}" STREQUAL "")
    string(TOLOWER "${_setting}" _option)
    string(REPLACE "_" "-" _option "${_option}")
    list(APPEND _library_arguments "--${_option}=${UDONCL_${_setting}}")
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

set(_device_argument)
if(DEFINED UDONCL_DEVICE_MODULE AND NOT "${UDONCL_DEVICE_MODULE}" STREQUAL "")
  set(_device_argument --device-llvm "${UDONCL_DEVICE_MODULE}")
endif()
set(_static_tensor_argument)
if(DEFINED UDONCL_STATIC_TENSOR_DIRECTORY AND
   NOT "${UDONCL_STATIC_TENSOR_DIRECTORY}" STREQUAL "")
  set(_static_tensor_argument --static-tensor-directory
      "${UDONCL_STATIC_TENSOR_DIRECTORY}")
endif()
set(_host_buffer_plan_argument)
if(DEFINED UDONCL_HOST_BUFFER_PLAN AND
   NOT "${UDONCL_HOST_BUFFER_PLAN}" STREQUAL "")
  set(_host_buffer_plan_argument --host-buffer-plan
      "${UDONCL_HOST_BUFFER_PLAN}")
endif()
set(_host_arguments)
if(DEFINED UDONCL_HOST_MLIR AND NOT "${UDONCL_HOST_MLIR}" STREQUAL "")
  list(APPEND _host_arguments --host-mlir "${UDONCL_HOST_MLIR}")
endif()
if(DEFINED UDONCL_HOST_MODULE AND NOT "${UDONCL_HOST_MODULE}" STREQUAL "")
  list(APPEND _host_arguments --host-llvm "${UDONCL_HOST_MODULE}")
endif()
if(DEFINED UDONCL_HOST_RUNTIME_MODULE AND
   NOT "${UDONCL_HOST_RUNTIME_MODULE}" STREQUAL "")
  list(APPEND _host_arguments --host-library-llvm
       "${UDONCL_HOST_RUNTIME_MODULE}")
endif()
file(REMOVE_RECURSE "${UDONCL_OUTPUT_DIRECTORY}")
execute_process(
  COMMAND "${UDONCL_COMPILER}" --emit-unity-library-bundle
          ${_library_arguments} ${_host_arguments} ${_device_argument} ${_static_tensor_argument}
          ${_host_buffer_plan_argument}
          ${_runtime_validation_argument} ${_intermediate_ir_argument}
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
