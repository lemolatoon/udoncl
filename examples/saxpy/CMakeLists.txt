cmake_minimum_required(VERSION 3.30...4.2)
project(udoncl_saxpy LANGUAGES NONE)

find_package(UdonCL 0.1 REQUIRED CONFIG)

# The Host C runs the OpenCL C kernel of saxpy.cl through the standard OpenCL
# API, which the linked UdonCL OpenCL runtime implements.
udoncl_add_host_library(saxpy_host host.c)
udoncl_add_device_library(saxpy_device saxpy.cl)

# The Host C names the program (UDONCL_PROGRAM), its exports (UDONCL_EXPORT)
# and its public buffers (UDONCL_PUBLIC). Unity imports only the Udon
# Assembly, the manifest and the shaders; leaving out the intermediate MLIR
# keeps the artifact small.
udoncl_add_unity_library(saxpy-unity-library
  OMIT_INTERMEDIATE_IR
  HOST_LIBRARIES saxpy_host UdonCL::OpenCLRuntime
  DEVICE_LIBRARIES saxpy_device)
