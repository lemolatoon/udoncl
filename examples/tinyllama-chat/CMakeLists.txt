cmake_minimum_required(VERSION 3.24)
project(udoncl_tinyllama_chat LANGUAGES NONE)

# A TinyLlama chat in one Host C file (tinyllama_chat.c) and one OpenCL C
# kernel (argmax.cl). The model and the tokenizer are the pinned TinyLlama
# files that scripts/fetch_tinyllama_onnx.py and
# scripts/fetch_tinyllama_tokenizer.py download and verify.
find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)
find_package(UdonCL 0.1 CONFIG REQUIRED)
find_package(UdonBLAS 0.1 CONFIG REQUIRED)
find_package(UdonDNN 0.1 CONFIG REQUIRED)
find_package(UdonATen 0.1 CONFIG REQUIRED)
find_package(UdonONNXOperators 0.1 CONFIG REQUIRED)
find_package(UdonONNXRuntime 0.1 CONFIG REQUIRED)
find_package(UdonTokenizers 0.1 CONFIG REQUIRED)

set(UDONCL_TINYLLAMA_ONNX_MODEL "" CACHE FILEPATH
  "The TinyLlama ONNX model (scripts/fetch_tinyllama_onnx.py)")
set(UDONCL_TINYLLAMA_TOKENIZER_DIRECTORY "" CACHE PATH
  "The TinyLlama tokenizer files (scripts/fetch_tinyllama_tokenizer.py)")
set(UDONCL_TINYLLAMA_CONTEXT_CAPACITY 2048 CACHE STRING
  "Tokens one conversation holds")
set(UDONCL_TINYLLAMA_MAX_NEW_TOKENS 32 CACHE STRING
  "Most tokens of one reply")
foreach(variable UDONCL_TINYLLAMA_ONNX_MODEL
                 UDONCL_TINYLLAMA_TOKENIZER_DIRECTORY)
  if(NOT ${variable} OR NOT EXISTS "${${variable}}")
    message(FATAL_ERROR "Set ${variable} to an existing path")
  endif()
endforeach()
set(capacity ${UDONCL_TINYLLAMA_CONTEXT_CAPACITY})
set(max_new_tokens ${UDONCL_TINYLLAMA_MAX_NEW_TOKENS})
if(NOT capacity MATCHES "^[1-9][0-9]*$" OR capacity LESS 2)
  message(FATAL_ERROR
    "UDONCL_TINYLLAMA_CONTEXT_CAPACITY must be an integer of at least 2")
endif()
if(NOT max_new_tokens MATCHES "^[1-9][0-9]*$" OR
   max_new_tokens GREATER capacity)
  message(FATAL_ERROR
    "UDONCL_TINYLLAMA_MAX_NEW_TOKENS must be a positive integer no larger "
    "than the capacity")
endif()

# The model runs one token per run (batch_size and sequence_length 1). Its
# past is a bounded dimension: each run binds it at the length the
# conversation reached, at most capacity - 1 rows.
math(EXPR past_length "${capacity} - 1")
udoncl_add_onnx_host_library(tinyllama_model
  MODEL "${UDONCL_TINYLLAMA_ONNX_MODEL}"
  DIMENSIONS batch_size=1 sequence_length=1
  DIMENSION_BOUNDS past_sequence_length=${past_length})
target_include_directories(tinyllama_model INTERFACE
  "${UdonBLAS_INCLUDE_DIRS}"
  "${UdonDNN_INCLUDE_DIRS}"
  "${UdonATen_INCLUDE_DIRS}")

# The Host C file, compiled next to the tokenizer tables (tinyllama_tokenizer.h)
# that are generated from the tokenizer files at build time.
udontokenizers_add_huggingface_model(tinyllama_chat_host
  TOKENIZER_DIRECTORY "${UDONCL_TINYLLAMA_TOKENIZER_DIRECTORY}"
  SYMBOL tinyllama_tokenizer
  SOURCES tinyllama_chat.c)
target_include_directories(tinyllama_chat_host INTERFACE
  "${UdonONNXRuntime_INCLUDE_DIRS}")
target_compile_definitions(tinyllama_chat_host INTERFACE
  TINYLLAMA_CAPACITY=${capacity}u
  TINYLLAMA_MAX_NEW_TOKENS=${max_new_tokens}u)

# The ArgMax kernel the Host launches by name.
udoncl_add_device_library(tinyllama_argmax argmax.cl)

# The Unity library. The real model stores its Device buffers and weights as
# four words per RGBA32 float texel.
udoncl_add_unity_library(tinyllama-chat
  DEVICE_TEXEL_FORMAT rgba32_float
  OMIT_INTERMEDIATE_IR
  HOST_LIBRARIES
    tinyllama_model
    tinyllama_chat_host
    UdonONNXRuntime::UdonONNXRuntime
    UdonTokenizers::UdonTokenizers
  DEVICE_LIBRARIES
    tinyllama_argmax
  LIBRARIES
    UdonONNXOperators::UdonONNXOperators
    UdonATen::UdonATen
    UdonBLAS::UdonBLAS)
