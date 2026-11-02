cmake_minimum_required(VERSION 3.30...4.2)
project(udoncl_saxpy LANGUAGES NONE)

find_package(UdonCL 0.1 REQUIRED CONFIG)

udoncl_add_host_library(saxpy_host host.c)
udoncl_add_device_library(saxpy_device unity_saxpy.cl)

udoncl_add_unity_library(saxpy-unity-library
  CONFIG unity-library.json
  HOST_LIBRARIES saxpy_host
  DEVICE_LIBRARIES saxpy_device)
