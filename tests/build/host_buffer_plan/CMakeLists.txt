cmake_minimum_required(VERSION 3.24)
project(udoncl_host_buffer_plan_test LANGUAGES NONE)

# A Unity library whose C Host allocates its Device buffers in constant-trip
# loops (host.c): the Host buffer plan gives the OpenCL runtime one buffer
# handle per allocation.
find_package(UdonCL 0.1 CONFIG REQUIRED)

udoncl_add_host_library(plan_host "${CMAKE_CURRENT_SOURCE_DIR}/host.c")
udoncl_add_unity_library(host_buffer_plan
  HOST_LIBRARIES plan_host UdonCL::OpenCLRuntime)
