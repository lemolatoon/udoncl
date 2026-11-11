cmake_minimum_required(VERSION 3.30...4.2)
project(udoncl_hello_kernel LANGUAGES NONE)

find_package(UdonCL 0.1 REQUIRED CONFIG)
find_package(UdonTokenizers 0.1 REQUIRED CONFIG)

udoncl_add_host_library(hello_host
  host.c
  host_support.c
  include/hello_host.h)
target_include_directories(hello_host INTERFACE include)
udoncl_add_host_library(hello_host_extra host_extra.c)
udoncl_add_host_library(hello_buffer_copy_host buffer_copy.c)
udoncl_add_host_library(hello_dispatch_host dispatch.c)
udoncl_add_host_library(hello_tokenizer_host tokenizer.c)
target_include_directories(hello_tokenizer_host INTERFACE
  "${UdonTokenizers_INCLUDE_DIRS}")
udoncl_add_host_library(hello_tokenizer_raw_host tokenizer_raw.c)
target_include_directories(hello_tokenizer_raw_host INTERFACE
  "${UdonTokenizers_INCLUDE_DIRS}")
udoncl_add_device_library(hello_device
  kernel.cl
  kernel_support.cl
  include/hello_device.h)
target_include_directories(hello_device INTERFACE include)
udoncl_add_device_library(hello_device_extra kernel_extra.cl)
udoncl_add_device_library(hello_device_int kernel_int.cl)

udoncl_add_artifact(hello-artifact
  CONFIG kernel.json
  HOST_LIBRARIES hello_host hello_host_extra
  DEVICE_LIBRARIES hello_device hello_device_extra)

udoncl_add_unity_library(hello-unity-library
  CONFIG unity-library.json
  HOST_LIBRARIES hello_host hello_host_extra
  DEVICE_LIBRARIES hello_device hello_device_extra)

udoncl_add_unity_library(hello-tokenizer-unity-library
  CONFIG tokenizer-library.json
  HOST_LIBRARIES hello_tokenizer_host UdonTokenizers::UdonTokenizers)

udoncl_add_unity_library(hello-tokenizer-raw-unity-library
  CONFIG tokenizer-raw-library.json
  HOST_LIBRARIES hello_tokenizer_raw_host UdonTokenizers::UdonTokenizers)

if(UDONCL_TINYLLAMA_TOKENIZER_DIRECTORY)
  udontokenizers_add_huggingface_model(tinyllama_tokenizer_model
    TOKENIZER_DIRECTORY "${UDONCL_TINYLLAMA_TOKENIZER_DIRECTORY}"
    SYMBOL tinyllama_tokenizer
    SOURCES tinyllama_tokenizer.c)
  udoncl_add_unity_library(tinyllama-tokenizer-unity-library
    CONFIG tinyllama-tokenizer-library.json
    HOST_LIBRARIES tinyllama_tokenizer_model UdonTokenizers::UdonTokenizers)
endif()

udoncl_add_unity_library(hello-buffer-copy-unity-library
  CONFIG buffer-copy-library.json
  HOST_LIBRARIES hello_buffer_copy_host)

udoncl_add_unity_library(hello-dispatch-unity-library
  CONFIG dispatch-library.json
  HOST_LIBRARIES hello_dispatch_host
  DEVICE_LIBRARIES hello_device hello_device_extra)

udoncl_add_unity_library(hello-int-dispatch-unity-library
  CONFIG dispatch-int-library.json
  HOST_LIBRARIES hello_dispatch_host
  DEVICE_LIBRARIES hello_device_int)
