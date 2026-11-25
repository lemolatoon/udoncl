/* A TinyLlama chat for VRChat in one Host C file.
 *
 * U# calls the exports through the generated binding
 * UdonclTinyllamaChatBinding:
 *
 *   Initialize          create the OpenCL objects and the model session
 *   BeginMessage(text)  start tokenizing the next user message
 *   ContinueMessage()   tokenize a bounded slice; 1 while work is pending,
 *                       0 once the message is ready, -status on failure
 *   Generate            run the message, then reply greedily until EOS or
 *                       TINYLLAMA_MAX_NEW_TOKENS
 *   ReadStreamedText    copy the reply text that is final so far, also
 *                       while Generate runs
 *   ReadGeneratedTokens copy the token IDs of the last reply
 *   ReadConversationTokens
 *                       copy the committed history of the conversation
 *   RestoreConversation(tokens, count)
 *                       continue the conversation of another Host from its
 *                       committed history
 *   Dispose             release everything; Initialize may run again
 *
 * The model runs one position per udononnxRun: its past is a bounded
 * dimension of at most TINYLLAMA_CAPACITY - 1 rows (CMake's
 * DIMENSION_BOUNDS), bound at the extent the conversation reached. Each
 * layer keeps its K and V in one buffer of the whole capacity, bound both as
 * the past (t rows) and as the present (t + 1 rows), so a run writes only
 * its own row (in-place Concat). A conversation grows: every message
 * continues after the previous reply, and Dispose + Initialize starts a new
 * one. RestoreConversation replaces it with the committed history of another
 * Host, so that players can take turns in one conversation: only its tokens
 * travel, and each Host rebuilds its KV cache from them.
 *
 * Every function stays within what udonclc lowers statically: each buffer
 * is created once in initialize, the loops over the layers have a constant
 * trip count, and each graph tensor is bound from one export. */
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdint.h>
#include <udoncl/array.h>
#include <udoncl/export.h>
#include <udoncl/string.h>
#include <udononnxruntime.h>
#include <udontokenizers.h>

#include <tinyllama_tokenizer.h>

/* Unity imports the library under Assets/udoncl-generated/tinyllama_chat. */
UDONCL_PROGRAM("tinyllama_chat");

/* Both come from CMake (UDONCL_TINYLLAMA_CONTEXT_CAPACITY and
 * UDONCL_TINYLLAMA_MAX_NEW_TOKENS), which bounds the model's past with the
 * same capacity. */
#ifndef TINYLLAMA_CAPACITY
#define TINYLLAMA_CAPACITY 2048u
#endif
#ifndef TINYLLAMA_MAX_NEW_TOKENS
#define TINYLLAMA_MAX_NEW_TOKENS 32u
#endif

#define VOCABULARY 32000u
#define LAYERS 22
/* One layer's K or V over the whole capacity: [1, 4, capacity, 64] floats. */
#define CACHE_BYTES ((size_t)TINYLLAMA_CAPACITY * 4u * 64u * sizeof(float))
/* Tokenizer work per ContinueMessage call: about one UTF-16 code unit of
 * the message and chat template, or one step of merging. */
#define MESSAGE_WORK 32
/* A token covers at most its piece length in symbols, so a message with
 * more symbols than this cannot fit the capacity either. */
#define MESSAGE_WORDS                                                          \
  UDONTOKENIZERS_ENCODER_WORKSPACE_WORDS(                                      \
      TINYLLAMA_CAPACITY *TINYLLAMA_TOKENIZER_MAX_PIECE_CODE_UNITS)
#define REPLY_UNITS                                                            \
  (TINYLLAMA_MAX_NEW_TOKENS * TINYLLAMA_TOKENIZER_MAX_PIECE_CODE_UNITS)

/* The ArgMax kernel is compiled ahead of time from argmax.cl, the Device
 * library of this example, so the program source is only a placeholder. */
static const char kernel_source[] = "argmax.cl";

/* The graph tensors, named as in the ONNX model. */
#define LAYER_NAMES(prefix, suffix)                                            \
  {prefix "0" suffix,  prefix "1" suffix,  prefix "2" suffix,                  \
   prefix "3" suffix,  prefix "4" suffix,  prefix "5" suffix,                  \
   prefix "6" suffix,  prefix "7" suffix,  prefix "8" suffix,                  \
   prefix "9" suffix,  prefix "10" suffix, prefix "11" suffix,                 \
   prefix "12" suffix, prefix "13" suffix, prefix "14" suffix,                 \
   prefix "15" suffix, prefix "16" suffix, prefix "17" suffix,                 \
   prefix "18" suffix, prefix "19" suffix, prefix "20" suffix,                 \
   prefix "21" suffix}
static const char *const past_key_names[LAYERS] =
    LAYER_NAMES("past_key_values.", ".key");
static const char *const past_value_names[LAYERS] =
    LAYER_NAMES("past_key_values.", ".value");
static const char *const present_key_names[LAYERS] =
    LAYER_NAMES("present.", ".key");
static const char *const present_value_names[LAYERS] =
    LAYER_NAMES("present.", ".value");
static const int64_t token_dims[2] = {1, 1};
static const int64_t logits_dims[3] = {1, 1, VOCABULARY};
static const int64_t mask_buffer_dims[2] = {1, TINYLLAMA_CAPACITY};
static const int64_t cache_buffer_dims[4] = {1, 4, TINYLLAMA_CAPACITY, 64};
static const int64_t mask_one = 1;

/* OpenCL objects and the session, from initialize to dispose. */
static cl_context context;
static cl_command_queue queue;
static udononnxSession_t *session;
static cl_mem input_ids, position_ids, attention_mask, logits, token;
static cl_mem keys[LAYERS], values[LAYERS];
static cl_program program;
static cl_kernel argmax;
static int initialized;

/* The conversation: the KV cache holds the positions [0, position), whose
 * tokens are staged_tokens[0, position) below. The last token of the
 * previous reply was selected but never run; the next message runs it
 * first, followed by EOS when the reply was cut off. Those closing tokens
 * after the KV cache's tokens are the committed history: a Host whose KV
 * cache holds exactly them continues the conversation the same way. */
static uint32_t position;
static int has_reply;
static uint32_t last_token;
/* Set when a generation failed; only Dispose + Initialize recover. */
static int failed;

/* The message being tokenized: the U# string of BeginMessage, the stage and
 * cursor of the chat template, and the encoder. Once ContinueMessage reports
 * 0 the message tokens are the first message_count words of the workspace. */
#define STAGE_IDLE 0
#define STAGE_FIRST 1
#define STAGE_TURN 2
#define STAGE_MERGE 3
static udon_string_t message;
static int32_t message_stage, message_cursor;
static udontokenizers_encoder_t encoder;
static int32_t message_words[MESSAGE_WORDS];
static uint32_t message_count;

/* The reply: its tokens and its text, decoded as the tokens arrive. */
static int32_t reply_tokens[TINYLLAMA_MAX_NEW_TOKENS];
static uint32_t reply_count;
static udontokenizers_decoder_t decoder;
static int32_t reply_units[REPLY_UNITS];
/* The token the ArgMax kernel selected, read back by the Host. */
static uint32_t selected;
/* Host sources of the non-blocking token and position uploads, indexed by
 * position, so a source stays intact until the queue has consumed it. */
static int64_t staged_tokens[TINYLLAMA_CAPACITY];
static int64_t staged_positions[TINYLLAMA_CAPACITY];

static udononnxStatus_t tokenizer_status(udontokenizersStatus_t status) {
  switch (status) {
  case UDONTOKENIZERS_STATUS_SUCCESS:
    return UDONONNX_STATUS_SUCCESS;
  case UDONTOKENIZERS_STATUS_OUTPUT_TOO_SMALL:
    return UDONONNX_STATUS_CAPACITY_EXCEEDED;
  case UDONTOKENIZERS_STATUS_INVALID_UTF16:
  case UDONTOKENIZERS_STATUS_INVALID_ARGUMENT:
    return UDONONNX_STATUS_INVALID_VALUE;
  default:
    return UDONONNX_STATUS_EXECUTION_FAILURE;
  }
}

/* The tensors whose shape never changes: the token, its position and the
 * logits of the one position a run computes. */
static udononnxStatus_t bind_fixed(void) {
  udononnxStatus_t status;
  status = udononnxBindInput(session, "input_ids", input_ids, 2, token_dims, 0);
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  status = udononnxBindInput(session, "position_ids", position_ids, 2,
                             token_dims, 0);
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  return udononnxBindOutput(session, "logits", logits, 3, logits_dims, 0);
}

/* The tensors of a run at position t: the mask of t + 1 ones, and each
 * layer's past of t rows and present of t + 1 rows, both the leading block
 * of the layer's buffer. */
static udononnxStatus_t bind_position(uint32_t t) {
  int64_t mask_dims[2] = {1, (int64_t)t + 1};
  int64_t past_dims[4] = {1, 4, (int64_t)t, 64};
  int64_t present_dims[4] = {1, 4, (int64_t)t + 1, 64};
  udononnxStatus_t status;
  status = udononnxBindInput(session, "attention_mask", attention_mask, 2,
                             mask_dims, mask_buffer_dims);
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  for (int layer = 0; layer < LAYERS; ++layer) {
    status = udononnxBindInput(session, past_key_names[layer], keys[layer], 4,
                               past_dims, cache_buffer_dims);
    if (status != UDONONNX_STATUS_SUCCESS)
      return status;
    status = udononnxBindInput(session, past_value_names[layer], values[layer],
                               4, past_dims, cache_buffer_dims);
    if (status != UDONONNX_STATUS_SUCCESS)
      return status;
    status = udononnxBindOutput(session, present_key_names[layer], keys[layer],
                                4, present_dims, cache_buffer_dims);
    if (status != UDONONNX_STATUS_SUCCESS)
      return status;
    status =
        udononnxBindOutput(session, present_value_names[layer], values[layer],
                           4, present_dims, cache_buffer_dims);
    if (status != UDONONNX_STATUS_SUCCESS)
      return status;
  }
  return UDONONNX_STATUS_SUCCESS;
}

/* Runs one token at the next position. */
static udononnxStatus_t run_token(uint32_t value) {
  udononnxStatus_t status;
  if (position >= TINYLLAMA_CAPACITY)
    return UDONONNX_STATUS_CAPACITY_EXCEEDED;
  staged_tokens[position] = value;
  staged_positions[position] = position;
  if (clEnqueueWriteBuffer(queue, input_ids, CL_FALSE, 0, sizeof(int64_t),
                           &staged_tokens[position], 0, 0, 0) != CL_SUCCESS ||
      clEnqueueWriteBuffer(queue, position_ids, CL_FALSE, 0, sizeof(int64_t),
                           &staged_positions[position], 0, 0, 0) != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  status = bind_position(position);
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  status = udononnxRun(session);
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  ++position;
  return UDONONNX_STATUS_SUCCESS;
}

/* Greedy selection: the ArgMax kernel of argmax.cl writes the index of the
 * largest logit into `token`, and the Host reads that one word back. */
static udononnxStatus_t select_token(void) {
  size_t one = 1;
  if (clSetKernelArg(argmax, 0, sizeof(cl_mem), &logits) != CL_SUCCESS ||
      clSetKernelArg(argmax, 1, sizeof(cl_mem), &token) != CL_SUCCESS ||
      clEnqueueNDRangeKernel(queue, argmax, 1, 0, &one, &one, 0, 0, 0) !=
          CL_SUCCESS ||
      clEnqueueReadBuffer(queue, token, CL_TRUE, 0, sizeof(selected), &selected,
                          0, 0, 0) != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  return selected < VOCABULARY ? UDONONNX_STATUS_SUCCESS
                               : UDONONNX_STATUS_EXECUTION_FAILURE;
}

UDONCL_EXPORT udononnxStatus_t initialize(void) {
  cl_platform_id platform = 0;
  cl_device_id device = 0;
  const char *source = kernel_source;
  cl_int error = CL_SUCCESS;
  udontokenizers_model_t model;
  udononnxStatus_t status;
  if (initialized)
    return UDONONNX_STATUS_INVALID_STATE;
  if (clGetPlatformIDs(1, &platform, 0) != CL_SUCCESS ||
      clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, 0) != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  context = clCreateContext(0, 1, &device, 0, 0, &error);
  if (error != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  queue = clCreateCommandQueue(context, device, 0, &error);
  if (error != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  input_ids =
      clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int64_t), 0, &error);
  if (error != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  position_ids =
      clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int64_t), 0, &error);
  if (error != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  attention_mask =
      clCreateBuffer(context, CL_MEM_READ_WRITE,
                     TINYLLAMA_CAPACITY * sizeof(int64_t), 0, &error);
  if (error != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  logits = clCreateBuffer(context, CL_MEM_READ_WRITE,
                          VOCABULARY * sizeof(float), 0, &error);
  if (error != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  token =
      clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(uint32_t), 0, &error);
  if (error != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  for (int layer = 0; layer < LAYERS; ++layer) {
    keys[layer] =
        clCreateBuffer(context, CL_MEM_READ_WRITE, CACHE_BYTES, 0, &error);
    if (error != CL_SUCCESS)
      return UDONONNX_STATUS_OPENCL_ERROR;
    values[layer] =
        clCreateBuffer(context, CL_MEM_READ_WRITE, CACHE_BYTES, 0, &error);
    if (error != CL_SUCCESS)
      return UDONONNX_STATUS_OPENCL_ERROR;
  }
  status = udononnxCreateSession(context, queue, &session);
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  status = bind_fixed();
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  /* Every leading block of the mask is all ones. */
  if (clEnqueueFillBuffer(queue, attention_mask, &mask_one, sizeof(mask_one), 0,
                          TINYLLAMA_CAPACITY * sizeof(int64_t), 0, 0,
                          0) != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  program = clCreateProgramWithSource(context, 1, &source, 0, &error);
  if (error != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  if (clBuildProgram(program, 0, 0, 0, 0, 0) != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  argmax = clCreateKernel(program, "tinyllama_argmax", &error);
  if (error != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  /* A new conversation with no reply yet. */
  tinyllama_tokenizerInitializeModel(&model);
  status = tokenizer_status(udontokenizersDecoderBeginRaw(
      &model, &decoder, reply_units, REPLY_UNITS, 1));
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  position = 0;
  has_reply = 0;
  failed = 0;
  message_stage = STAGE_IDLE;
  message_count = 0;
  reply_count = 0;
  initialized = 1;
  return UDONONNX_STATUS_SUCCESS;
}

/* Starts tokenizing `text` as the next user message: the first message of
 * a conversation is BOS + the chat template's user turn + the generation
 * prompt; a later one continues after the previous reply. A message that is
 * ready but was not generated yet is replaced. */
UDONCL_EXPORT udononnxStatus_t begin_message(udon_string_t text) {
  udontokenizers_model_t model;
  udononnxStatus_t status;
  int first;
  if (!initialized)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  if (failed)
    return UDONONNX_STATUS_INVALID_STATE;
  message_stage = STAGE_IDLE;
  message_count = 0;
  tinyllama_tokenizerInitializeModel(&model);
  /* Only the first message starts with BOS. After a restore the reply is
   * already in the KV cache. */
  first = !has_reply && position == 0;
  status = tokenizer_status(udontokenizersEncoderBeginRaw(
      &model, &encoder, message_words, MESSAGE_WORDS, first));
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  message = text;
  message_cursor = 0;
  message_stage = first ? STAGE_FIRST : STAGE_TURN;
  return UDONONNX_STATUS_SUCCESS;
}

/* Does at most MESSAGE_WORK units of tokenizer work on the begun message.
 * Returns 1 while work is pending, 0 once the message is ready for Generate
 * and -status on failure, which abandons the message. A message that does
 * not fit the rest of the capacity fails with CAPACITY_EXCEEDED. */
UDONCL_EXPORT int32_t continue_message(void) {
  udontokenizers_model_t model;
  udontokenizers_chat_template_t chat;
  udontokenizersStatus_t status = UDONTOKENIZERS_STATUS_SUCCESS;
  int32_t remaining = MESSAGE_WORK;
  int32_t finished = 0;
  uint32_t closing;
  if (!initialized)
    return -(int32_t)UDONONNX_STATUS_NOT_INITIALIZED;
  if (message_stage == STAGE_IDLE)
    return -(int32_t)UDONONNX_STATUS_INVALID_STATE;
  tinyllama_tokenizerInitializeModel(&model);
  tinyllama_tokenizerInitializeChatTemplate(&chat);
  /* The chat template around the message, then the generation prompt. */
  while (remaining > 0 && message_stage != STAGE_MERGE) {
    int32_t before = message_cursor;
    if (message_stage == STAGE_FIRST)
      status = udontokenizersEncoderAppendChatMessagePartRaw(
          &model, &chat, &encoder, UDONTOKENIZERS_CHAT_ROLE_USER, message,
          &message_cursor, remaining, &finished, message_words);
    else
      status = udontokenizersEncoderAppendChatTurnPartRaw(
          &model, &chat, &encoder, UDONTOKENIZERS_CHAT_ROLE_ASSISTANT,
          UDONTOKENIZERS_CHAT_ROLE_USER, message, &message_cursor, remaining,
          &finished, message_words);
    if (status != UDONTOKENIZERS_STATUS_SUCCESS)
      break;
    remaining -= message_cursor - before;
    if (finished) {
      status = udontokenizersEncoderAppendChatGenerationPromptRaw(
          &model, &chat, &encoder, message_words);
      message_stage = STAGE_MERGE;
      if (status != UDONTOKENIZERS_STATUS_SUCCESS)
        break;
    }
  }
  /* Merging the symbols into tokens. */
  finished = 0;
  if (status == UDONTOKENIZERS_STATUS_SUCCESS && remaining > 0)
    status = udontokenizersEncoderStepRaw(&model, &encoder, message_words,
                                          remaining, &finished);
  if (status != UDONTOKENIZERS_STATUS_SUCCESS) {
    message_stage = STAGE_IDLE;
    return -(int32_t)tokenizer_status(status);
  }
  if (!finished)
    return 1;
  message_stage = STAGE_IDLE;
  /* The tokens that close the previous reply run before the message, and
   * every run position stays below the capacity. */
  closing =
      has_reply ? (last_token == (uint32_t)model.eos_token_id ? 1u : 2u) : 0u;
  if (encoder.count <= 0 || closing > TINYLLAMA_CAPACITY - position ||
      (uint32_t)encoder.count > TINYLLAMA_CAPACITY - position - closing)
    return -(int32_t)UDONONNX_STATUS_CAPACITY_EXCEEDED;
  message_count = (uint32_t)encoder.count;
  return 0;
}

/* A generation that failed leaves the conversation unusable. */
static udononnxStatus_t fail(udononnxStatus_t status) {
  failed = 1;
  return status;
}

/* Runs the ready message, one token per run, then replies: the ArgMax of
 * each run's logits is the next token, which runs in turn until EOS or the
 * reply limit. The limit is TINYLLAMA_MAX_NEW_TOKENS or what the capacity
 * still holds. The last token is left for the next message to run. */
UDONCL_EXPORT udononnxStatus_t generate(void) {
  udontokenizers_model_t model;
  udononnxStatus_t status;
  uint32_t count = message_count, index, limit;
  int32_t length = 0;
  if (!initialized)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  if (failed || !count || count > TINYLLAMA_CAPACITY)
    return UDONONNX_STATUS_INVALID_STATE;
  message_count = 0;
  reply_count = 0;
  tinyllama_tokenizerInitializeModel(&model);
  status = tokenizer_status(udontokenizersDecoderBeginRaw(
      &model, &decoder, reply_units, REPLY_UNITS, 1));
  if (status != UDONONNX_STATUS_SUCCESS)
    return fail(status);
  /* Close the previous reply: its last token, and EOS when it was cut
   * off. */
  if (has_reply) {
    has_reply = 0;
    status = run_token(last_token);
    if (status != UDONONNX_STATUS_SUCCESS)
      return fail(status);
    if (last_token != (uint32_t)model.eos_token_id) {
      status = run_token((uint32_t)model.eos_token_id);
      if (status != UDONONNX_STATUS_SUCCESS)
        return fail(status);
    }
  }
  for (index = 0; index < count; ++index) {
    status = run_token((uint32_t)message_words[index]);
    if (status != UDONONNX_STATUS_SUCCESS)
      return fail(status);
  }
  limit = TINYLLAMA_CAPACITY + 1u - position;
  if (limit > TINYLLAMA_MAX_NEW_TOKENS)
    limit = TINYLLAMA_MAX_NEW_TOKENS;
  for (index = 0; index < limit; ++index) {
    /* The token selected last runs before the next one is selected. */
    if (index > 0) {
      status = run_token(selected);
      if (status != UDONONNX_STATUS_SUCCESS)
        return fail(status);
    }
    status = select_token();
    if (status != UDONONNX_STATUS_SUCCESS)
      return fail(status);
    reply_tokens[index] = (int32_t)selected;
    reply_count = index + 1u;
    status = tokenizer_status(udontokenizersDecoderAppendRaw(
        &model, &decoder, (int32_t)selected, reply_units));
    if (status != UDONONNX_STATUS_SUCCESS)
      return fail(status);
    if (selected == (uint32_t)model.eos_token_id)
      break;
  }
  status = tokenizer_status(
      udontokenizersDecoderFinishRaw(&model, &decoder, reply_units, &length));
  if (status != UDONONNX_STATUS_SUCCESS)
    return fail(status);
  last_token = selected;
  has_reply = 1;
  return UDONONNX_STATUS_SUCCESS;
}

/* Copies the final UTF-16 code units of the reply from `offset` on to the
 * start of `destination`, at most its length. Returns the number copied or
 * -status. The text only grows while Generate runs and is the whole reply
 * once it completed; special tokens such as EOS are skipped. */
UDONCL_EXPORT_WHILE_BUSY int32_t
read_streamed_text(udon_int32_array_t destination, uint32_t offset) {
  int32_t count, index;
  if (!initialized)
    return -(int32_t)UDONONNX_STATUS_NOT_INITIALIZED;
  if (offset > (uint32_t)decoder.final_count)
    return -(int32_t)UDONONNX_STATUS_INVALID_VALUE;
  count = decoder.final_count - (int32_t)offset;
  if (count > udonInt32ArrayLength(destination))
    count = udonInt32ArrayLength(destination);
  for (index = 0; index < count; ++index)
    udonInt32ArraySet(destination, index, reply_units[(int32_t)offset + index]);
  return count;
}

/* Copies the token IDs of the last reply to `destination`. Returns their
 * number or -status. */
UDONCL_EXPORT int32_t read_generated_tokens(udon_int32_array_t destination) {
  uint32_t count = reply_count, index;
  if (!initialized)
    return -(int32_t)UDONONNX_STATUS_NOT_INITIALIZED;
  if (count > TINYLLAMA_MAX_NEW_TOKENS)
    return -(int32_t)UDONONNX_STATUS_EXECUTION_FAILURE;
  if (udonInt32ArrayLength(destination) < (int32_t)count)
    return -(int32_t)UDONONNX_STATUS_INVALID_VALUE;
  for (index = 0; index < count; ++index)
    udonInt32ArraySet(destination, (int32_t)index, reply_tokens[index]);
  return (int32_t)count;
}

/* Copies the committed history of the conversation to `destination`: the
 * tokens of the KV cache, then the tokens that close the last reply (its
 * last token, and EOS when it was cut off). RestoreConversation continues
 * the conversation from it on another Host. Returns the number of tokens or
 * -status; a history beyond the capacity, which no message could follow,
 * fails with CAPACITY_EXCEEDED. */
UDONCL_EXPORT int32_t read_conversation_tokens(udon_int32_array_t destination) {
  udontokenizers_model_t model;
  uint32_t count = position, closing = 0, index;
  if (!initialized)
    return -(int32_t)UDONONNX_STATUS_NOT_INITIALIZED;
  if (failed)
    return -(int32_t)UDONONNX_STATUS_INVALID_STATE;
  if (count > TINYLLAMA_CAPACITY)
    return -(int32_t)UDONONNX_STATUS_EXECUTION_FAILURE;
  tinyllama_tokenizerInitializeModel(&model);
  if (has_reply)
    closing = last_token == (uint32_t)model.eos_token_id ? 1u : 2u;
  if (closing > TINYLLAMA_CAPACITY - count)
    return -(int32_t)UDONONNX_STATUS_CAPACITY_EXCEEDED;
  if (udonInt32ArrayLength(destination) < (int32_t)(count + closing))
    return -(int32_t)UDONONNX_STATUS_INVALID_VALUE;
  for (index = 0; index < count; ++index)
    udonInt32ArraySet(destination, (int32_t)index,
                      (int32_t)staged_tokens[index]);
  if (closing > 0)
    udonInt32ArraySet(destination, (int32_t)count, (int32_t)last_token);
  if (closing > 1)
    udonInt32ArraySet(destination, (int32_t)count + 1, model.eos_token_id);
  return (int32_t)(count + closing);
}

/* Replaces the conversation with the committed history tokens[0, count)
 * that ReadConversationTokens copied, usually on another Host. The KV cache
 * keeps the leading tokens it already holds in the same order; the rest run
 * one token per run. Count 0 starts a new conversation. A history beyond
 * the capacity fails with CAPACITY_EXCEEDED and a token outside the
 * vocabulary with INVALID_VALUE, both leaving the conversation unchanged.
 * Otherwise a message that was not generated yet and the last reply's text
 * and tokens are dropped. */
UDONCL_EXPORT udononnxStatus_t restore_conversation(udon_int32_array_t tokens,
                                                    int32_t count) {
  udontokenizers_model_t model;
  udononnxStatus_t status;
  uint32_t total, kept, index;
  if (!initialized)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  if (failed)
    return UDONONNX_STATUS_INVALID_STATE;
  if (count < 0)
    return UDONONNX_STATUS_INVALID_VALUE;
  if ((uint32_t)count > TINYLLAMA_CAPACITY)
    return UDONONNX_STATUS_CAPACITY_EXCEEDED;
  if (count > udonInt32ArrayLength(tokens))
    return UDONONNX_STATUS_INVALID_VALUE;
  total = (uint32_t)count;
  for (index = 0; index < total; ++index) {
    int32_t value = udonInt32ArrayGet(tokens, (int32_t)index);
    if (value < 0 || value >= (int32_t)VOCABULARY)
      return UDONONNX_STATUS_INVALID_VALUE;
  }
  /* The common prefix with the KV cache stays. */
  kept = position < total ? position : total;
  for (index = 0; index < kept; ++index)
    if (staged_tokens[index] != udonInt32ArrayGet(tokens, (int32_t)index))
      break;
  kept = index;
  tinyllama_tokenizerInitializeModel(&model);
  status = tokenizer_status(udontokenizersDecoderBeginRaw(
      &model, &decoder, reply_units, REPLY_UNITS, 1));
  if (status != UDONONNX_STATUS_SUCCESS)
    return fail(status);
  message_stage = STAGE_IDLE;
  message_count = 0;
  reply_count = 0;
  has_reply = 0;
  position = kept;
  /* A run may suspend the export until a later frame: the rest of the
   * history is staged first, so `tokens` is only read before the first run. */
  for (index = kept; index < total; ++index)
    staged_tokens[index] = udonInt32ArrayGet(tokens, (int32_t)index);
  for (index = kept; index < total; ++index) {
    status = run_token((uint32_t)staged_tokens[index]);
    if (status != UDONONNX_STATUS_SUCCESS)
      return fail(status);
  }
  return UDONONNX_STATUS_SUCCESS;
}

/* Releases everything initialize created. Initialize may run again, which
 * starts a new conversation. */
UDONCL_EXPORT udononnxStatus_t dispose(void) {
  udononnxStatus_t status;
  if (!initialized)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  status = udononnxReleaseSession(session);
  clReleaseKernel(argmax);
  clReleaseProgram(program);
  for (int layer = 0; layer < LAYERS; ++layer) {
    clReleaseMemObject(keys[layer]);
    clReleaseMemObject(values[layer]);
  }
  clReleaseMemObject(token);
  clReleaseMemObject(logits);
  clReleaseMemObject(attention_mask);
  clReleaseMemObject(position_ids);
  clReleaseMemObject(input_ids);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  message_stage = STAGE_IDLE;
  message_count = 0;
  initialized = 0;
  return status;
}
