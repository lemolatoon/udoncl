include_guard(GLOBAL)

# One prefix owns both packages. A repository-local SDK takes precedence over
# system paths and stale LLVM_DIR/MLIR_DIR entries from an earlier configure.
set(VRC_LLVM_ROOT "" CACHE PATH "LLVM/MLIR install prefix (overrides repository-local discovery)")
set(vrc_llvm_prefix "${VRC_LLVM_ROOT}")
if(NOT vrc_llvm_prefix AND EXISTS "${PROJECT_SOURCE_DIR}/.deps/llvm23")
  set(vrc_llvm_prefix "${PROJECT_SOURCE_DIR}/.deps/llvm23")
endif()

if(vrc_llvm_prefix)
  cmake_path(ABSOLUTE_PATH vrc_llvm_prefix
    BASE_DIRECTORY "${PROJECT_SOURCE_DIR}" NORMALIZE)
  foreach(package IN ITEMS LLVM MLIR)
    string(TOLOWER "${package}" package_dir)
    set(config_dir "${vrc_llvm_prefix}/lib/cmake/${package_dir}")
    if(NOT EXISTS "${config_dir}/${package}Config.cmake")
      message(FATAL_ERROR
        "Missing ${package}Config.cmake under ${config_dir}. "
        "Complete the local SDK installation or set VRC_LLVM_ROOT to a complete LLVM/MLIR prefix.")
    endif()
    set(${package}_DIR "${config_dir}" CACHE PATH "${package} package configuration" FORCE)
  endforeach()
  find_package(LLVM ${VRC_LLVM_VERSION} EXACT REQUIRED CONFIG
    PATHS "${LLVM_DIR}" NO_DEFAULT_PATH)
  find_package(MLIR REQUIRED CONFIG PATHS "${MLIR_DIR}" NO_DEFAULT_PATH)
else()
  find_package(LLVM ${VRC_LLVM_VERSION} EXACT REQUIRED CONFIG)
  # Even on the system fallback path, MLIR must belong to the selected LLVM.
  set(MLIR_DIR "${LLVM_INSTALL_PREFIX}/lib/cmake/mlir"
    CACHE PATH "MLIR package configuration" FORCE)
  find_package(MLIR REQUIRED CONFIG PATHS "${MLIR_DIR}" NO_DEFAULT_PATH)
endif()

if(NOT LLVM_PACKAGE_VERSION VERSION_EQUAL VRC_LLVM_VERSION)
  message(FATAL_ERROR
    "udoncl requires LLVM/MLIR ${VRC_LLVM_VERSION}; found ${LLVM_PACKAGE_VERSION}")
endif()
message(STATUS "udoncl LLVM ${LLVM_PACKAGE_VERSION}: ${LLVM_DIR}")
message(STATUS "udoncl MLIR: ${MLIR_DIR}")
