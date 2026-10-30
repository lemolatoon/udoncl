include_guard(GLOBAL)

# Usage requirements shared by project targets, including MLIR's object targets.
add_library(vrc_build_options INTERFACE)
add_library(VRC::BuildOptions ALIAS vrc_build_options)
target_compile_features(vrc_build_options INTERFACE cxx_std_17)
target_include_directories(vrc_build_options INTERFACE
  "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
  "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>"
)
# LLVM's exported targets do not carry all standalone include/definition
# requirements. Keep that adapter target-scoped rather than changing directories.
target_include_directories(vrc_build_options SYSTEM INTERFACE
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)
separate_arguments(vrc_llvm_definitions NATIVE_COMMAND "${LLVM_DEFINITIONS}")
target_compile_definitions(vrc_build_options INTERFACE ${vrc_llvm_definitions})
target_compile_options(vrc_build_options INTERFACE
  "$<$<CXX_COMPILER_ID:MSVC>:/bigobj>"
  "$<$<CXX_COMPILER_ID:MSVC>:/utf-8>"
)

function(vrc_configure_target target)
  target_link_libraries(${target} PUBLIC VRC::BuildOptions)
  set_target_properties(${target} PROPERTIES CXX_EXTENSIONS OFF)
  # AddMLIR may compile through an object library. Give it the same usage
  # requirements explicitly; do not rely on directory-scope inherited flags.
  if(TARGET obj.${target})
    target_link_libraries(obj.${target} PRIVATE VRC::BuildOptions)
    set_target_properties(obj.${target} PROPERTIES CXX_EXTENSIONS OFF)
  endif()
endfunction()

# TableGen is a build tool and does not inherit C++ target include directories.
set(VRC_TABLEGEN_INCLUDES "${PROJECT_SOURCE_DIR}/include" ${MLIR_INCLUDE_DIRS})
