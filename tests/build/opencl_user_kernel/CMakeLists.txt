cmake_minimum_required(VERSION 3.30...4.2)
project(udoncl_opencl_user_kernel LANGUAGES NONE)

find_package(UdonCL 0.1 REQUIRED CONFIG)

# A Host C written against the standard OpenCL API and the OpenCL AOT
# runtime, with OpenCL C kernels of its own: the Unity library config holds
# only public names and buffers.
udoncl_add_host_library(user_kernel_host host.c)
udoncl_add_device_library(user_kernel_device kernels.cl)
udoncl_add_unity_library(opencl-user-kernel
  CONFIG unity-library.json
  HOST_LIBRARIES user_kernel_host UdonCL::OpenCLRuntime
  DEVICE_LIBRARIES user_kernel_device)

# The same kernel launched over a run-time global work size, which the build
# cannot derive.
udoncl_add_host_library(user_kernel_dynamic_host dynamic.c)
udoncl_add_unity_library(opencl-user-kernel-dynamic
  CONFIG dynamic-library.json
  HOST_LIBRARIES user_kernel_dynamic_host UdonCL::OpenCLRuntime
  DEVICE_LIBRARIES user_kernel_device)
