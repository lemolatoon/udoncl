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
  # The real model stores its Device buffers and weights as four words per
  # RGBA32 float texel (see docs/architecture/
  # udon-vrchat-shader-constraints.md); rgba8_unorm restores one word per
  # RGBA8 texel. The synthetic fixture keeps the default.
  set(UDONCL_TINYLLAMA_DEVICE_TEXEL_FORMAT "rgba32_float" CACHE STRING
    "Device texel format of the real TinyLlama artifacts")
  string(JSON real_unity_config SET "${real_unity_config}"
    device_texel_format "\"${UDONCL_TINYLLAMA_DEVICE_TEXEL_FORMAT}\"")
  string(JSON real_unity_config SET "${real_unity_config}"
    program "\"tinyllama_real_prefill\"")
  string(JSON real_unity_config SET "${real_unity_config}"
    binding class_name "\"UdonclTinyLlamaPrefillBinding\"")
  set(unity_config "${CMAKE_CURRENT_BINARY_DIR}/unity-library-real.json")
  file(WRITE "${unity_config}" "${real_unity_config}\n")

  # Context Capacity is a build parameter: the M=1 graph is specialized to
  # past_sequence_length = capacity - 1 and a capacity-long attention mask,
  # and the C Host receives the same bound. Generic kernel limits, not this
  # file, decide whether a capacity compiles.
  set(UDONCL_TINYLLAMA_CONTEXT_CAPACITY 512 CACHE STRING
    "TinyLlama conversation Context Capacity in tokens")
  # The 4096-token Conversation and Chat Artifacts (the -4k targets) have a
  # Host file of their own, conversation_host_4k.c, and are configured next
  # to the ones above. Their U# API, program and binding names are the same,
  # so a project imports one of the two.
  option(UDONCL_TINYLLAMA_CONTEXT_4K
    "Also configure the 4096-token TinyLlama Conversation and Chat Artifacts"
    ON)
  set(tinyllama_context_4k 4096)
  set(UDONCL_TINYLLAMA_MAX_NEW_TOKENS 32 CACHE STRING
    "Maximum new tokens one TinyLlama generation may request")
  if(NOT UDONCL_TINYLLAMA_CONTEXT_CAPACITY MATCHES "^[1-9][0-9]*$" OR
     UDONCL_TINYLLAMA_CONTEXT_CAPACITY LESS 2)
    message(FATAL_ERROR
      "UDONCL_TINYLLAMA_CONTEXT_CAPACITY must be an integer of at least 2")
  endif()
  if(NOT UDONCL_TINYLLAMA_MAX_NEW_TOKENS MATCHES "^[1-9][0-9]*$" OR
     UDONCL_TINYLLAMA_MAX_NEW_TOKENS GREATER UDONCL_TINYLLAMA_CONTEXT_CAPACITY)
    message(FATAL_ERROR
      "UDONCL_TINYLLAMA_MAX_NEW_TOKENS must be a positive integer no larger "
      "than the Context Capacity")
  endif()
  # Positions one model run of the Conversation and Chat Artifacts covers.
  # The graph is specialized to sequence_length = rows on top of the same
  # past, so a prompt runs in chunks of this many positions; a run that fills
  # fewer rows (the last chunk, and every decode step) pads the rest and
  # commits only its real rows (udon_onnx_binding_enqueue_advance_sequence_state_rows).
  # 1 runs every position on its own.
  set(UDONCL_TINYLLAMA_SEQUENCE_ROWS 1 CACHE STRING
    "Positions one TinyLlama Conversation/Chat model run covers")
  if(NOT UDONCL_TINYLLAMA_SEQUENCE_ROWS MATCHES "^[1-9][0-9]*$" OR
     UDONCL_TINYLLAMA_SEQUENCE_ROWS GREATER UDONCL_TINYLLAMA_CONTEXT_CAPACITY)
    message(FATAL_ERROR
      "UDONCL_TINYLLAMA_SEQUENCE_ROWS must be a positive integer no larger "
      "than the Context Capacity")
  endif()
  set(conversation_unity_config_json "${real_unity_config}")
  string(JSON conversation_unity_config_json SET
    "${conversation_unity_config_json}" program
    "\"tinyllama_real_conversation\"")
  string(JSON conversation_unity_config_json SET
    "${conversation_unity_config_json}" binding class_name
    "\"UdonclTinyLlamaConversationBinding\"")
  string(JSON conversation_unity_config_json SET
    "${conversation_unity_config_json}" generate_onnx_public_buffers false)
  set(conversation_exports
    [[{"symbol":"tinyllama_initialize","public_name":"Initialize","parameters":[]}]]
    [[{"symbol":"tinyllama_reset","public_name":"Reset","parameters":[]}]]
    [[{"symbol":"tinyllama_set_prompt_tokens","public_name":"SetPromptTokens","parameters":["tokenIds","tokenCount"]}]]
    [[{"symbol":"tinyllama_generate","public_name":"Generate","parameters":["maxNewTokens"]}]]
    [[{"symbol":"tinyllama_generated_token_count","public_name":"GeneratedTokenCount","parameters":[]}]]
    [[{"symbol":"tinyllama_read_generated_tokens","public_name":"ReadGeneratedTokens","parameters":["destination"]}]]
    [[{"symbol":"tinyllama_dispose","public_name":"Shutdown","parameters":[]}]]
  )
  foreach(export RANGE 0 6)
    list(GET conversation_exports ${export} export_json)
    string(JSON conversation_unity_config_json SET
      "${conversation_unity_config_json}" export_overrides ${export}
      "${export_json}")
  endforeach()
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

  # Chat Artifact: the same token-array Conversation Host plus the optional
  # text layer (chat template, special tokens and UTF-16 decoding). The
  # Conversation Artifact above stays free of tokenizer code. A prompt can be
  # tokenized in bounded slices (BeginChatPrompt, then ContinueChatPrompt
  # until ChatPromptPending is 0), and the reply text is streamed: the
  # StreamedTextLength code units read by ReadStreamedText are final. After a
  # reply, BeginChatTurn/ContinueChatTurn append the next user message to the
  # same conversation and ResetChat starts a new one.
  set(chat_unity_config_json "${conversation_unity_config_json}")
  string(JSON chat_unity_config_json SET "${chat_unity_config_json}"
    program "\"tinyllama_real_chat\"")
  string(JSON chat_unity_config_json SET "${chat_unity_config_json}"
    binding class_name "\"UdonclTinyLlamaChatBinding\"")
  set(chat_exports
    [[{"symbol":"tinyllama_set_chat_prompt","public_name":"SetChatPrompt","parameters":["systemText","userText"]}]]
    [[{"symbol":"tinyllama_begin_chat_prompt","public_name":"BeginChatPrompt","parameters":[]}]]
    [[{"symbol":"tinyllama_continue_chat_prompt","public_name":"ContinueChatPrompt","parameters":["systemText","userText","maxWork"]}]]
    [[{"symbol":"tinyllama_chat_prompt_pending","public_name":"ChatPromptPending","parameters":[]}]]
    [[{"symbol":"tinyllama_begin_chat_turn","public_name":"BeginChatTurn","parameters":[]}]]
    [[{"symbol":"tinyllama_continue_chat_turn","public_name":"ContinueChatTurn","parameters":["userText","maxWork"]}]]
    [[{"symbol":"tinyllama_reset_chat","public_name":"ResetChat","parameters":[]}]]
    [[{"symbol":"tinyllama_prompt_token_count","public_name":"PromptTokenCount","parameters":[]}]]
    [[{"symbol":"tinyllama_read_prompt_tokens","public_name":"ReadPromptTokens","parameters":["destination"]}]]
    [[{"symbol":"tinyllama_generated_text_length","public_name":"GeneratedTextLength","parameters":[]}]]
    [[{"symbol":"tinyllama_read_generated_text","public_name":"ReadGeneratedText","parameters":["destination"]}]]
    [[{"symbol":"tinyllama_streamed_text_length","public_name":"StreamedTextLength","parameters":[],"allowed_while_busy":true}]]
    [[{"symbol":"tinyllama_read_streamed_text","public_name":"ReadStreamedText","parameters":["destination","offset","count"],"allowed_while_busy":true}]]
    [[{"symbol":"tinyllama_last_tokenizer_status","public_name":"LastTokenizerStatus","parameters":[]}]]
  )
  string(JSON chat_export_count LENGTH "${chat_unity_config_json}"
    export_overrides)
  foreach(export_json IN LISTS chat_exports)
    string(JSON chat_unity_config_json SET "${chat_unity_config_json}"
      export_overrides ${chat_export_count} "${export_json}")
    math(EXPR chat_export_count "${chat_export_count} + 1")
  endforeach()
  set(chat_unity_config "${CMAKE_CURRENT_BINARY_DIR}/unity-library-chat.json")
  file(WRITE "${chat_unity_config}" "${chat_unity_config_json}\n")

  # Acceptance-only M=1 step validation artifact: the same capacity graph
  # as the Conversation Artifact, but with public input/output buffers and the
  # plain initialize/run Host, so a Unity test can feed one ORT step (including
  # its rolling past K/V) and compare every logit and present K/V output.
  set(step_unity_config_json "${real_unity_config}")
  string(JSON step_unity_config_json SET "${step_unity_config_json}"
    program "\"tinyllama_real_m1_step\"")
  string(JSON step_unity_config_json SET "${step_unity_config_json}"
    binding class_name "\"UdonclTinyLlamaStepBinding\"")
  set(step_unity_config "${CMAKE_CURRENT_BINARY_DIR}/unity-library-step.json")
  file(WRITE "${step_unity_config}" "${step_unity_config_json}\n")
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
# The graph of `rows` positions per run on a past of capacity - 1 rows
# (past_sequence_length = capacity - 1, an attention mask of
# capacity + rows - 1), as the ONNX Host library `target`.
function(tinyllama_add_model_library target capacity rows)
  math(EXPR past_length "${capacity} - 1")
  math(EXPR window "${capacity} + ${rows} - 1")
  udoncl_add_onnx_host_library(${target}
    MODEL "${model}"
    DIMENSION_BOUNDS
      batch_size=1
      sequence_length=${rows}
      past_sequence_length=${past_length}
    INPUT_SHAPES
      attention_mask=1,${window})
  target_include_directories(${target}
    INTERFACE
      "${UdonBLAS_INCLUDE_DIRS}"
      "${UdonDNN_INCLUDE_DIRS}"
      "${UdonATen_INCLUDE_DIRS}")
endfunction()

# The Conversation Host libraries of one Context Capacity, named with
# `suffix`: the graph of UDONCL_TINYLLAMA_SEQUENCE_ROWS positions per run
# (the M=1 graph by default) specialized to past_sequence_length =
# capacity - 1, the token-array C Host `host` and, with a tokenizer, the chat
# text layer, all compiled with the same capacity and rows.
function(tinyllama_add_conversation_libraries suffix capacity host)
  set(rows ${UDONCL_TINYLLAMA_SEQUENCE_ROWS})
  tinyllama_add_model_library(transformer_conversation_model_host${suffix}
    ${capacity} ${rows})
  set(definitions
    TINYLLAMA_CONTEXT_CAPACITY=${capacity}u
    TINYLLAMA_GENERATED_CAPACITY=${UDONCL_TINYLLAMA_MAX_NEW_TOKENS}u
    TINYLLAMA_SEQUENCE_ROWS=${rows}u)
  udoncl_add_host_library(transformer_conversation_user_host${suffix} ${host})
  target_include_directories(transformer_conversation_user_host${suffix}
    INTERFACE "${UdonONNXRuntime_INCLUDE_DIRS}")
  target_compile_definitions(transformer_conversation_user_host${suffix}
    INTERFACE ${definitions})
  if(UDONCL_TINYLLAMA_TOKENIZER_DIRECTORY)
    udontokenizers_add_huggingface_model(transformer_chat_tokenizer${suffix}
      TOKENIZER_DIRECTORY "${UDONCL_TINYLLAMA_TOKENIZER_DIRECTORY}"
      SYMBOL tinyllama_tokenizer
      SOURCES conversation_text.c)
    target_include_directories(transformer_chat_tokenizer${suffix} INTERFACE
      "${UdonONNXRuntime_INCLUDE_DIRS}")
    target_compile_definitions(transformer_chat_tokenizer${suffix} INTERFACE
      ${definitions})
  endif()
endfunction()

# The Conversation Artifact, and with a tokenizer the Chat Artifact, of the
# libraries named with `suffix`; the artifact targets end in `target_suffix`.
# An optional JSON file of further `device_kernels` adds the library kernels
# only that capacity dispatches: a declared kernel is compiled into the
# artifact whether or not it runs.
function(tinyllama_add_conversation_artifacts suffix target_suffix)
  set(conversation_config "${conversation_unity_config}")
  set(chat_config "${chat_unity_config}")
  if(ARGC GREATER 2)
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/${ARGV2}" extra_json)
    string(JSON extra_count LENGTH "${extra_json}" device_kernels)
    math(EXPR extra_last "${extra_count} - 1")
    foreach(kind IN ITEMS conversation chat)
      set(config_json "${${kind}_unity_config_json}")
      string(JSON kernel_count LENGTH "${config_json}" device_kernels)
      foreach(extra RANGE ${extra_last})
        string(JSON kernel GET "${extra_json}" device_kernels ${extra})
        string(JSON config_json SET "${config_json}" device_kernels
          ${kernel_count} "${kernel}")
        math(EXPR kernel_count "${kernel_count} + 1")
      endforeach()
      set(${kind}_config
        "${CMAKE_CURRENT_BINARY_DIR}/unity-library-${kind}${target_suffix}.json")
      file(WRITE "${${kind}_config}" "${config_json}\n")
    endforeach()
  endif()
  udoncl_add_unity_library(onnx-transformer-tiny-conversation${target_suffix}
    CONFIG "${conversation_config}"
    OMIT_INTERMEDIATE_IR
    HOST_LIBRARIES
      transformer_conversation_model_host${suffix}
      transformer_conversation_user_host${suffix}
      UdonONNXRuntime::UdonONNXRuntime
    LIBRARIES
      UdonONNXOperators::UdonONNXOperators
      UdonATen::UdonATen
      UdonBLAS::UdonBLAS)
  if(UDONCL_TINYLLAMA_TOKENIZER_DIRECTORY)
    udoncl_add_unity_library(onnx-transformer-tiny-chat${target_suffix}
      CONFIG "${chat_config}"
      OMIT_INTERMEDIATE_IR
      HOST_LIBRARIES
        transformer_conversation_model_host${suffix}
        transformer_conversation_user_host${suffix}
        transformer_chat_tokenizer${suffix}
        UdonONNXRuntime::UdonONNXRuntime
        UdonTokenizers::UdonTokenizers
      LIBRARIES
        UdonONNXOperators::UdonONNXOperators
        UdonATen::UdonATen
        UdonBLAS::UdonBLAS)
  endif()
endfunction()

if(UDONCL_TINYLLAMA_ONNX_MODEL AND UDONCL_TINYLLAMA_TOKENIZER_DIRECTORY)
  find_package(UdonTokenizers 0.1 CONFIG REQUIRED)
endif()
if(UDONCL_TINYLLAMA_ONNX_MODEL)
  tinyllama_add_conversation_libraries(""
    ${UDONCL_TINYLLAMA_CONTEXT_CAPACITY} conversation_host.c)
  if(UDONCL_TINYLLAMA_CONTEXT_4K)
    tinyllama_add_conversation_libraries(_4k ${tinyllama_context_4k}
      conversation_host_4k.c)
  endif()
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
  if(UDONCL_TINYLLAMA_SEQUENCE_ROWS EQUAL 1)
    tinyllama_add_conversation_artifacts("" "")
  else()
    # The attention products of several rows per run take the large
    # strided-batched SGEMM, which the 4096-token artifacts declare too.
    tinyllama_add_conversation_artifacts("" "" unity-library-4k-kernels.json)
  endif()
  if(UDONCL_TINYLLAMA_CONTEXT_4K)
    tinyllama_add_conversation_artifacts(_4k -4k unity-library-4k-kernels.json)
  endif()
  # The M=1 step validation artifact keeps the M=1 graph of the Context
  # Capacity whatever rows the Conversation Artifacts run.
  set(step_model_host transformer_conversation_model_host)
  if(NOT UDONCL_TINYLLAMA_SEQUENCE_ROWS EQUAL 1)
    set(step_model_host transformer_step_model_host)
    tinyllama_add_model_library(${step_model_host}
      ${UDONCL_TINYLLAMA_CONTEXT_CAPACITY} 1)
  endif()
  udoncl_add_unity_library(onnx-transformer-tiny-m1-step
    CONFIG "${step_unity_config}"
    OMIT_INTERMEDIATE_IR
    HOST_LIBRARIES
      ${step_model_host}
      transformer_user_host
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
