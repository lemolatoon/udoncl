cmake_minimum_required(VERSION 3.30...4.2)
project(udonblas_sgemm_example LANGUAGES NONE)

find_package(UdonCL 0.1 REQUIRED CONFIG)
find_package(UdonBLAS 0.1 REQUIRED CONFIG)

udoncl_add_host_library(udonblas_sgemm_example_host host.c)
target_include_directories(udonblas_sgemm_example_host INTERFACE
  "${UdonBLAS_INCLUDE_DIRS}")

udoncl_add_unity_library(udonblas-sgemm-example
  CONFIG unity-library.json
  HOST_LIBRARIES udonblas_sgemm_example_host
  LIBRARIES UdonBLAS::UdonBLAS)
