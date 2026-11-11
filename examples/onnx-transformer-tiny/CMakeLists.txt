cmake_minimum_required(VERSION 3.24)
project(udoncl_onnx_transformer_tiny LANGUAGES NONE)

find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)
find_package(UdonCL 0.1 CONFIG REQUIRED)
find_package(UdonBLAS 0.1 CONFIG REQUIRED)
find_package(UdonDNN 0.1 CONFIG REQUIRED)
find_package(UdonATen 0.1 CONFIG REQUIRED)
find_package(UdonONNXOperators 0.1 CONFIG REQUIRED)
find_package(UdonONNXRuntime 0.1 CONFIG REQUIRED)

set(UDONCL_TINYLLAMA_ONNX_MODEL "" CACHE FILEPATH
  "Use an external TinyLlama-compatible ONNX model instead of the fast synthetic fixture")
set(UDONCL_TINYLLAMA_TOKENIZER_DIRECTORY "" CACHE PATH
  "Compile the pinned TinyLlama tokenizer tables for conversation Host input")
if(UDONCL_TINYLLAMA_ONNX_MODEL)
  cmake_path(ABSOLUTE_PATH UDONCL_TINYLLAMA_ONNX_MODEL NORMALIZE
             OUTPUT_VARIABLE model)
  if(NOT EXISTS "${model}")
    message(FATAL_ERROR "TinyLlama ONNX model does not exist: ${model}")
  endif()
  set(model_shape_arguments
    DIMENSION_BOUNDS
      batch_size=1
      sequence_length=4
      past_sequence_length=0
    INPUT_SHAPES
      attention_mask=1,4)
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/unity-library.json" real_unity_config)
  string(JSON real_unity_config SET "${real_unity_config}"
    program "\"tinyllama_real_prefill\"")
  string(JSON real_unity_config SET "${real_unity_config}"
    binding class_name "\"UdonclTinyLlamaPrefillBinding\"")
  set(unity_config "${CMAKE_CURRENT_BINARY_DIR}/unity-library-real.json")
  file(WRITE "${unity_config}" "${real_unity_config}\n")

  set(conversation_shape_arguments
    DIMENSION_BOUNDS
      batch_size=1
      sequence_length=1
      past_sequence_length=63
    INPUT_SHAPES
      attention_mask=1,64)
  set(conversation_unity_config_json "${real_unity_config}")
  string(JSON conversation_unity_config_json SET
    "${conversation_unity_config_json}" program
    "\"tinyllama_real_conversation\"")
  string(JSON conversation_unity_config_json SET
    "${conversation_unity_config_json}" binding class_name
    "\"UdonclTinyLlamaConversationBinding\"")
  string(JSON conversation_unity_config_json SET
    "${conversation_unity_config_json}" generate_onnx_public_buffers false)
  string(JSON conversation_unity_config_json SET
    "${conversation_unity_config_json}" export_overrides 0 symbol
    "\"tinyllama_conversation_initialize\"")
  string(JSON conversation_unity_config_json SET
    "${conversation_unity_config_json}" export_overrides 1 symbol
    "\"tinyllama_conversation_run\"")
  string(JSON conversation_unity_config_json SET
    "${conversation_unity_config_json}" export_overrides 2 symbol
    "\"tinyllama_conversation_dispose\"")
  string(JSON conversation_unity_config_json SET
    "${conversation_unity_config_json}" export_overrides 2 public_name
    "\"Shutdown\"")
  string(JSON conversation_unity_config_json REMOVE
    "${conversation_unity_config_json}" export_overrides 3)
  string(JSON internal_export_count LENGTH
    "${conversation_unity_config_json}" internal_exports)
  string(JSON conversation_unity_config_json SET
    "${conversation_unity_config_json}" internal_exports
    ${internal_export_count}
    "\"udon_onnx_binding_enqueue_advance_sequence_state\"")
  math(EXPR internal_export_count "${internal_export_count} + 1")
  string(JSON conversation_unity_config_json SET
    "${conversation_unity_config_json}" internal_exports
    ${internal_export_count}
    "\"udon_onnx_binding_enqueue_reset_sequence_state\"")
  set(conversation_unity_config
    "${CMAKE_CURRENT_BINARY_DIR}/unity-library-conversation.json")
  file(WRITE "${conversation_unity_config}"
    "${conversation_unity_config_json}\n")
else()
  set(model "${CMAKE_CURRENT_BINARY_DIR}/transformer.onnx")
  execute_process(
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/generate_onnx.py"
            -o "${model}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr)
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "Transformer ONNX generation failed:\n${stdout}${stderr}")
  endif()
  set(model_shape_arguments DIMENSION_BOUNDS sequence=4)
  set(unity_config "${CMAKE_CURRENT_SOURCE_DIR}/unity-library.json")
endif()

udoncl_add_onnx_host_library(transformer_host MODEL "${model}"
  ${model_shape_arguments})
target_include_directories(transformer_host INTERFACE
  "${UdonBLAS_INCLUDE_DIRS}"
  "${UdonDNN_INCLUDE_DIRS}"
  "${UdonATen_INCLUDE_DIRS}")
udoncl_add_host_library(transformer_user_host host.c)
target_include_directories(transformer_user_host INTERFACE
  "${UdonONNXRuntime_INCLUDE_DIRS}")
if(UDONCL_TINYLLAMA_ONNX_MODEL)
  udoncl_add_onnx_host_library(transformer_conversation_model_host
    MODEL "${model}"
    ${conversation_shape_arguments})
  target_include_directories(transformer_conversation_model_host INTERFACE
    "${UdonBLAS_INCLUDE_DIRS}"
    "${UdonDNN_INCLUDE_DIRS}"
    "${UdonATen_INCLUDE_DIRS}")
  udoncl_add_host_library(transformer_conversation_user_host
    conversation_host.c)
  target_include_directories(transformer_conversation_user_host INTERFACE
    "${UdonONNXRuntime_INCLUDE_DIRS}")
endif()
if(UDONCL_TINYLLAMA_TOKENIZER_DIRECTORY)
  find_package(UdonTokenizers 0.1 CONFIG REQUIRED)
  udoncl_add_host_library(transformer_conversation_input_host
    conversation_input.c)
  target_include_directories(transformer_conversation_input_host INTERFACE
    "${UdonONNXRuntime_INCLUDE_DIRS}"
    "${UdonTokenizers_INCLUDE_DIRS}")
  udontokenizers_add_huggingface_model(transformer_conversation_tokenizer
    TOKENIZER_DIRECTORY "${UDONCL_TINYLLAMA_TOKENIZER_DIRECTORY}"
    SYMBOL tinyllama_tokenizer
    SOURCES conversation_tokenizer.c)
endif()
set(unity_library_arguments)
if(UDONCL_TINYLLAMA_ONNX_MODEL)
  list(APPEND unity_library_arguments OMIT_INTERMEDIATE_IR)
endif()
udoncl_add_unity_library(onnx-transformer-tiny
  CONFIG "${unity_config}"
  ${unity_library_arguments}
  HOST_LIBRARIES
    transformer_host
    transformer_user_host
    UdonONNXRuntime::UdonONNXRuntime
  LIBRARIES
    UdonONNXOperators::UdonONNXOperators
    UdonATen::UdonATen
    UdonBLAS::UdonBLAS)

if(UDONCL_TINYLLAMA_ONNX_MODEL)
  udoncl_add_unity_library(onnx-transformer-tiny-conversation
    CONFIG "${conversation_unity_config}"
    OMIT_INTERMEDIATE_IR
    HOST_LIBRARIES
      transformer_conversation_model_host
      transformer_conversation_user_host
      UdonONNXRuntime::UdonONNXRuntime
    LIBRARIES
      UdonONNXOperators::UdonONNXOperators
      UdonATen::UdonATen
      UdonBLAS::UdonBLAS)
endif()

# Keep tokenizer/input preparation out of the prefill validation artifact. The
# final Conversation Artifact is a separate target; this small synthetic target
# only proves that the same C helpers and generated tables lower to Udon.
if(UDONCL_TINYLLAMA_TOKENIZER_DIRECTORY AND
   NOT UDONCL_TINYLLAMA_ONNX_MODEL)
  udoncl_add_unity_library(onnx-transformer-conversation-input
    CONFIG "${unity_config}"
    HOST_LIBRARIES
      transformer_host
      transformer_user_host
      transformer_conversation_input_host
      transformer_conversation_tokenizer
      UdonONNXRuntime::UdonONNXRuntime
      UdonTokenizers::UdonTokenizers
    LIBRARIES
      UdonONNXOperators::UdonONNXOperators
      UdonATen::UdonATen
      UdonBLAS::UdonBLAS)
endif()
