cmake_minimum_required(VERSION 3.30...4.2)
project(udoncl_average_pooling LANGUAGES NONE)

find_package(UdonCL 0.1 REQUIRED CONFIG)

udoncl_add_host_library(average_pooling_host host.c)
udoncl_add_device_library(average_pooling_device average_pooling.cl)

udoncl_add_unity_library(average-pooling-unity-library
  CONFIG unity-library.json
  HOST_LIBRARIES average_pooling_host
  DEVICE_LIBRARIES average_pooling_device)
