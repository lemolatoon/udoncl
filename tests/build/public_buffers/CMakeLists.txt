cmake_minimum_required(VERSION 3.24)
project(udoncl_public_buffers_test LANGUAGES NONE)

# A SAXPY Unity library whose public Device buffers are UDONCL_PUBLIC Host
# variables (host.c), which also names the program.
find_package(UdonCL 0.1 CONFIG REQUIRED)

udoncl_add_host_library(public_host "${CMAKE_CURRENT_SOURCE_DIR}/host.c")
udoncl_add_device_library(saxpy_device "${CMAKE_CURRENT_SOURCE_DIR}/saxpy.cl")
udoncl_add_unity_library(public_buffers
  HOST_LIBRARIES public_host UdonCL::OpenCLRuntime
  DEVICE_LIBRARIES saxpy_device)
