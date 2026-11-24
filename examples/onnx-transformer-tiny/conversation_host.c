#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdint.h>
#include <udoncl/array.h>
#include <udoncl/export.h>
#include <udononnxruntime.h>

#include "conversation_host.h"

/* The program of the Conversation Artifact. A wrapper that includes this
 * file for another artifact (conversation_host_4k.c, chat_host.c) defines
 * its own TINYLLAMA_PROGRAM first. */
#ifndef TINYLLAMA_PROGRAM
#define TINYLLAMA_PROGRAM "tinyllama_real_conversation"
#endif
UDONCL_PROGRAM(TINYLLAMA_PROGRAM);

#define TINYLLAMA_LOGIT_WORDS 32000
#define TINYLLAMA_ARGMAX_PARTIAL_WORDS 512u
#define TINYLLAMA_ARGMAX_FAILURE 0xffffffffu
#define TINYLLAMA_EOS_TOKEN 2u
#define TINYLLAMA_LAYERS 22
/* One layer's K or V over the whole Context Capacity: [1, 4, capacity, 64]
 * floats. */
#define TINYLLAMA_CACHE_BYTES                                                  \
  ((size_t)TINYLLAMA_CONTEXT_CAPACITY * 4u * 64u * sizeof(float))

/* Private backend entry points. The public ATen ArgMax contract also needs
 * dim/keepdim and an int64 result, so this uint32 reduction stays private. */
cl_int __vrc_enqueue_udonaten_argmax_reduce_device_args(cl_command_queue,
                                                        cl_mem, cl_mem, cl_int,
                                                        cl_int, cl_uint,
                                                        const cl_event *,
                                                        cl_event *);
cl_int __vrc_enqueue_udonaten_argmax_finalize_device_args(
    cl_command_queue, cl_mem, cl_mem, cl_mem, cl_int, cl_int, cl_int, cl_uint,
    const cl_event *, cl_event *);

/* The model library is the M=1 graph with a bounded past
 * (DIMENSION_BOUNDS past_sequence_length = capacity - 1): one run is one
 * position on a past of `position` rows, and it costs as many rows as the
 * conversation holds.
 *
 * The Host owns every graph input and output and binds them through the
 * UdonONNX session API. Each layer keeps its K and V in one buffer of the
 * Context Capacity, bound as the past (the first `position` rows) and as the
 * present (the first position + 1 rows) of every run: the past is the
 * leading block of the present, so a run writes only its own row (in-place
 * Concat). Nothing is ever cleared; a reset only starts again at position 0,
 * and a later turn continues at the position the conversation reached. The
 * attention mask holds ones over the whole capacity, so its leading block is
 * right for every position. */
#define TINYLLAMA_LAYER_NAMES(prefix, suffix)                                  \
  {prefix "0" suffix,  prefix "1" suffix,  prefix "2" suffix,                  \
   prefix "3" suffix,  prefix "4" suffix,  prefix "5" suffix,                  \
   prefix "6" suffix,  prefix "7" suffix,  prefix "8" suffix,                  \
   prefix "9" suffix,  prefix "10" suffix, prefix "11" suffix,                 \
   prefix "12" suffix, prefix "13" suffix, prefix "14" suffix,                 \
   prefix "15" suffix, prefix "16" suffix, prefix "17" suffix,                 \
   prefix "18" suffix, prefix "19" suffix, prefix "20" suffix,                 \
   prefix "21" suffix}
static const char *const past_key_names[TINYLLAMA_LAYERS] =
    TINYLLAMA_LAYER_NAMES("past_key_values.", ".key");
static const char *const past_value_names[TINYLLAMA_LAYERS] =
    TINYLLAMA_LAYER_NAMES("past_key_values.", ".value");
static const char *const present_key_names[TINYLLAMA_LAYERS] =
    TINYLLAMA_LAYER_NAMES("present.", ".key");
static const char *const present_value_names[TINYLLAMA_LAYERS] =
    TINYLLAMA_LAYER_NAMES("present.", ".value");
static const int64_t token_dims[2] = {1, 1};
static const int64_t logits_dims[3] = {1, 1, TINYLLAMA_LOGIT_WORDS};
static const int64_t mask_buffer_dims[2] = {1, TINYLLAMA_CONTEXT_CAPACITY};
static const int64_t cache_buffer_dims[4] = {1, 4, TINYLLAMA_CONTEXT_CAPACITY,
                                             64};
static const int64_t mask_one = 1;

static cl_context context;
static cl_command_queue queue;
static udononnxSession_t *session;
static cl_mem input_ids, attention_mask, position_ids, logits;
/* One variable per layer's K and V buffer, reached through constant tables
 * of their addresses: udonclc unrolls the loops over the layers and folds
 * every table entry to its variable. */
#define TINYLLAMA_LAYER_VARIABLES(prefix)                                      \
  prefix##0, prefix##1, prefix##2, prefix##3, prefix##4, prefix##5, prefix##6, \
      prefix##7, prefix##8, prefix##9, prefix##10, prefix##11, prefix##12,     \
      prefix##13, prefix##14, prefix##15, prefix##16, prefix##17, prefix##18,  \
      prefix##19, prefix##20, prefix##21
#define TINYLLAMA_LAYER_ADDRESSES(prefix)                                      \
  {                                                                            \
      &prefix##0,  &prefix##1,  &prefix##2,  &prefix##3,  &prefix##4,          \
      &prefix##5,  &prefix##6,  &prefix##7,  &prefix##8,  &prefix##9,          \
      &prefix##10, &prefix##11, &prefix##12, &prefix##13, &prefix##14,         \
      &prefix##15, &prefix##16, &prefix##17, &prefix##18, &prefix##19,         \
      &prefix##20, &prefix##21}
static cl_mem TINYLLAMA_LAYER_VARIABLES(key_cache_);
static cl_mem TINYLLAMA_LAYER_VARIABLES(value_cache_);
static cl_mem *const keys[TINYLLAMA_LAYERS] =
    TINYLLAMA_LAYER_ADDRESSES(key_cache_);
static cl_mem *const values[TINYLLAMA_LAYERS] =
    TINYLLAMA_LAYER_ADDRESSES(value_cache_);
static cl_mem partials, result;
static int32_t prompt[TINYLLAMA_CONTEXT_CAPACITY];
static int32_t generated[TINYLLAMA_GENERATED_CAPACITY];
/* Host sources of the non-blocking uploads, indexed by position, so an
 * upload's source stays intact until the queue has consumed it. */
static int64_t staged_tokens[TINYLLAMA_CONTEXT_CAPACITY];
static int64_t staged_positions[TINYLLAMA_CONTEXT_CAPACITY];
/* Destination and event of the one outstanding selected-token read. */
static uint32_t selected_token;
static cl_event selected_event;
static uint32_t prompt_count, generated_count, generation_serial;
static int initialized, disposed, busy, completed, failed;
/* Multi-turn state. The KV cache holds the positions [0, context_count).
 * After a completed generation the last generated token was never run when
 * the generation reached its limit (last_unrun); a reply that ended with EOS
 * before the limit had its EOS run speculatively. A turn prepared after a
 * completed generation fills prompt[] with the tokens that close the
 * previous reply (closing_count of them) followed by the next turn; once
 * committed (continuing) the next generation feeds them from
 * context_count. */
static uint32_t context_count, closing_count;
static int last_unrun, turn_prepared, continuing;

static void clear_conversation(void) {
  prompt_count = generated_count = context_count = closing_count = 0;
  completed = failed = last_unrun = turn_prepared = continuing = 0;
}

#define TINYLLAMA_RELEASE(memory)                                              \
  do {                                                                         \
    if ((memory) && clReleaseMemObject(memory) != CL_SUCCESS)                  \
      failed_release = 1;                                                      \
    (memory) = 0;                                                              \
  } while (0)

static udononnxStatus_t release_opencl(void) {
  int failed_release = 0, layer;
  if (selected_event && clReleaseEvent(selected_event) != CL_SUCCESS)
    failed_release = 1;
  selected_event = 0;
  TINYLLAMA_RELEASE(result);
  TINYLLAMA_RELEASE(partials);
  for (layer = 0; layer < TINYLLAMA_LAYERS; ++layer) {
    TINYLLAMA_RELEASE(*keys[layer]);
    TINYLLAMA_RELEASE(*values[layer]);
  }
  TINYLLAMA_RELEASE(logits);
  TINYLLAMA_RELEASE(position_ids);
  TINYLLAMA_RELEASE(attention_mask);
  TINYLLAMA_RELEASE(input_ids);
  if (queue && clReleaseCommandQueue(queue) != CL_SUCCESS)
    failed_release = 1;
  queue = 0;
  if (context && clReleaseContext(context) != CL_SUCCESS)
    failed_release = 1;
  context = 0;
  return failed_release ? UDONONNX_STATUS_OPENCL_ERROR
                        : UDONONNX_STATUS_SUCCESS;
}

static udononnxStatus_t release_session(void) {
  udononnxStatus_t status = UDONONNX_STATUS_SUCCESS;
  if (session)
    status = udononnxReleaseSession(session);
  session = 0;
  return status;
}

static udononnxStatus_t cleanup(udononnxStatus_t status) {
  udononnxStatus_t released = release_session();
  udononnxStatus_t opencl = release_opencl();
  initialized = 0;
  if (released != UDONONNX_STATUS_SUCCESS)
    return released;
  return opencl == UDONONNX_STATUS_SUCCESS ? status : opencl;
}

static cl_mem create_buffer(size_t bytes, cl_int *error) {
  return clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, 0, error);
}

/* Binds the inputs and outputs whose shape does not change: the token and
 * its position, and the logits of the one position a run computes. */
static udononnxStatus_t bind_fixed(void) {
  udononnxStatus_t status =
      udononnxBindInput(session, "input_ids", input_ids, 2, token_dims, 0);
  if (status == UDONONNX_STATUS_SUCCESS)
    status = udononnxBindInput(session, "position_ids", position_ids, 2,
                               token_dims, 0);
  if (status == UDONONNX_STATUS_SUCCESS)
    status = udononnxBindOutput(session, "logits", logits, 3, logits_dims, 0);
  return status;
}

/* Binds the tensors of a run at `position`: the attention mask of
 * position + 1 ones, and each layer's past of `position` rows and present
 * of position + 1 rows, both the leading block of the layer's cache. Bind
 * copies the dimensions, so every run binds them again. */
static udononnxStatus_t bind_position(uint32_t position) {
  int64_t mask_dims[2] = {1, (int64_t)position + 1};
  int64_t past_dims[4] = {1, 4, (int64_t)position, 64};
  int64_t present_dims[4] = {1, 4, (int64_t)position + 1, 64};
  udononnxStatus_t status;
  int layer;
  status = udononnxBindInput(session, "attention_mask", attention_mask, 2,
                             mask_dims, mask_buffer_dims);
  for (layer = 0; layer < TINYLLAMA_LAYERS; ++layer) {
    if (status == UDONONNX_STATUS_SUCCESS)
      status = udononnxBindInput(session, past_key_names[layer], *keys[layer],
                                 4, past_dims, cache_buffer_dims);
    if (status == UDONONNX_STATUS_SUCCESS)
      status =
          udononnxBindInput(session, past_value_names[layer], *values[layer], 4,
                            past_dims, cache_buffer_dims);
    if (status == UDONONNX_STATUS_SUCCESS)
      status =
          udononnxBindOutput(session, present_key_names[layer], *keys[layer], 4,
                             present_dims, cache_buffer_dims);
    if (status == UDONONNX_STATUS_SUCCESS)
      status = udononnxBindOutput(session, present_value_names[layer],
                                  *values[layer], 4, present_dims,
                                  cache_buffer_dims);
  }
  return status;
}

/* Enqueues the run of the token already in input_ids at `position`. */
static udononnxStatus_t enqueue_run(uint32_t position) {
  udononnxStatus_t status;
  if (position >= TINYLLAMA_CONTEXT_CAPACITY)
    return UDONONNX_STATUS_CAPACITY_EXCEEDED;
  status = bind_position(position);
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  return udononnxRun(session);
}

UDONCL_EXPORT udononnxStatus_t initialize(void) {
  cl_platform_id platform = 0;
  cl_device_id device = 0;
  cl_int error = CL_SUCCESS;
  udononnxStatus_t status;
  int layer;
  if (disposed)
    return UDONONNX_STATUS_DISPOSED;
  if (initialized)
    return UDONONNX_STATUS_INVALID_STATE;
  if (clGetPlatformIDs(1, &platform, 0) != CL_SUCCESS ||
      clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, 0) != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  context = clCreateContext(0, 1, &device, 0, 0, &error);
  if (!context || error != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  queue = clCreateCommandQueue(context, device, 0, &error);
  if (!queue || error != CL_SUCCESS) {
    release_opencl();
    return UDONONNX_STATUS_OPENCL_ERROR;
  }
  input_ids = create_buffer(sizeof(int64_t), &error);
  if (!input_ids || error != CL_SUCCESS)
    return cleanup(UDONONNX_STATUS_OPENCL_ERROR);
  attention_mask =
      create_buffer(TINYLLAMA_CONTEXT_CAPACITY * sizeof(int64_t), &error);
  if (!attention_mask || error != CL_SUCCESS)
    return cleanup(UDONONNX_STATUS_OPENCL_ERROR);
  position_ids = create_buffer(sizeof(int64_t), &error);
  if (!position_ids || error != CL_SUCCESS)
    return cleanup(UDONONNX_STATUS_OPENCL_ERROR);
  logits = create_buffer(TINYLLAMA_LOGIT_WORDS * sizeof(float), &error);
  if (!logits || error != CL_SUCCESS)
    return cleanup(UDONONNX_STATUS_OPENCL_ERROR);
  /* One allocation call site per buffer: the Host buffer plan sizes the
   * handle and byte pools by counting allocation call sites. */
#define TINYLLAMA_CREATE_LAYER_CACHE(n)                                        \
  key_cache_##n = create_buffer(TINYLLAMA_CACHE_BYTES, &error);                \
  value_cache_##n = create_buffer(TINYLLAMA_CACHE_BYTES, &error)
  TINYLLAMA_CREATE_LAYER_CACHE(0);
  TINYLLAMA_CREATE_LAYER_CACHE(1);
  TINYLLAMA_CREATE_LAYER_CACHE(2);
  TINYLLAMA_CREATE_LAYER_CACHE(3);
  TINYLLAMA_CREATE_LAYER_CACHE(4);
  TINYLLAMA_CREATE_LAYER_CACHE(5);
  TINYLLAMA_CREATE_LAYER_CACHE(6);
  TINYLLAMA_CREATE_LAYER_CACHE(7);
  TINYLLAMA_CREATE_LAYER_CACHE(8);
  TINYLLAMA_CREATE_LAYER_CACHE(9);
  TINYLLAMA_CREATE_LAYER_CACHE(10);
  TINYLLAMA_CREATE_LAYER_CACHE(11);
  TINYLLAMA_CREATE_LAYER_CACHE(12);
  TINYLLAMA_CREATE_LAYER_CACHE(13);
  TINYLLAMA_CREATE_LAYER_CACHE(14);
  TINYLLAMA_CREATE_LAYER_CACHE(15);
  TINYLLAMA_CREATE_LAYER_CACHE(16);
  TINYLLAMA_CREATE_LAYER_CACHE(17);
  TINYLLAMA_CREATE_LAYER_CACHE(18);
  TINYLLAMA_CREATE_LAYER_CACHE(19);
  TINYLLAMA_CREATE_LAYER_CACHE(20);
  TINYLLAMA_CREATE_LAYER_CACHE(21);
#undef TINYLLAMA_CREATE_LAYER_CACHE
  for (layer = 0; layer < TINYLLAMA_LAYERS; ++layer)
    if (!*keys[layer] || !*values[layer])
      return cleanup(UDONONNX_STATUS_OPENCL_ERROR);
  partials =
      create_buffer(TINYLLAMA_ARGMAX_PARTIAL_WORDS * sizeof(uint32_t), &error);
  if (!partials || error != CL_SUCCESS)
    return cleanup(UDONONNX_STATUS_OPENCL_ERROR);
  result = create_buffer(sizeof(uint32_t), &error);
  if (!result || error != CL_SUCCESS)
    return cleanup(UDONONNX_STATUS_OPENCL_ERROR);
  status = udononnxCreateSession(context, queue, &session);
  if (status == UDONONNX_STATUS_SUCCESS)
    status = bind_fixed();
  if (status != UDONONNX_STATUS_SUCCESS)
    return cleanup(status);
  /* Every leading block of the mask is all ones. */
  if (clEnqueueFillBuffer(queue, attention_mask, &mask_one, sizeof(mask_one), 0,
                          TINYLLAMA_CONTEXT_CAPACITY * sizeof(int64_t), 0, 0,
                          0) != CL_SUCCESS)
    return cleanup(UDONONNX_STATUS_OPENCL_ERROR);
  clear_conversation();
  ++generation_serial;
  initialized = 1;
  return UDONONNX_STATUS_SUCCESS;
}

UDONCL_EXPORT udononnxStatus_t reset(void) {
  return tinyllamaConversationReset();
}

/* A new conversation starts at position 0 again; the rows the cache holds
 * past it are never read before a run overwrites them. */
udononnxStatus_t tinyllamaConversationReset(void) {
  if (disposed)
    return UDONONNX_STATUS_DISPOSED;
  if (!initialized)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  if (busy)
    return UDONONNX_STATUS_BUSY;
  if (clFinish(queue) != CL_SUCCESS) {
    failed = 1;
    return UDONONNX_STATUS_OPENCL_ERROR;
  }
  ++generation_serial;
  clear_conversation();
  return UDONONNX_STATUS_SUCCESS;
}

static udononnxStatus_t prompt_replacement_state(void) {
  if (disposed)
    return UDONONNX_STATUS_DISPOSED;
  if (!initialized)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  if (busy)
    return UDONONNX_STATUS_BUSY;
  if (failed)
    return UDONONNX_STATUS_INVALID_STATE;
  return UDONONNX_STATUS_SUCCESS;
}

UDONCL_EXPORT udononnxStatus_t set_prompt_tokens(udon_int32_array_t token_ids,
                                                 uint32_t token_count) {
  uint32_t index;
  udononnxStatus_t status = prompt_replacement_state();
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  if (!token_ids || !token_count ||
      udonInt32ArrayLength(token_ids) < (int32_t)token_count)
    return UDONONNX_STATUS_INVALID_VALUE;
  if (token_count > TINYLLAMA_CONTEXT_CAPACITY)
    return UDONONNX_STATUS_CAPACITY_EXCEEDED;
  if (generated_count || completed)
    return UDONONNX_STATUS_INVALID_STATE;
  for (index = 0; index < token_count; ++index) {
    int32_t token = udonInt32ArrayGet(token_ids, (int32_t)index);
    if (token < 0 || token >= TINYLLAMA_LOGIT_WORDS)
      return UDONONNX_STATUS_INVALID_VALUE;
  }
  for (index = 0; index < token_count; ++index)
    prompt[index] = udonInt32ArrayGet(token_ids, (int32_t)index);
  prompt_count = token_count;
  return UDONONNX_STATUS_SUCCESS;
}

udononnxStatus_t tinyllamaConversationPreparePrompt(int32_t **workspace) {
  udononnxStatus_t status = prompt_replacement_state();
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  if (!workspace)
    return UDONONNX_STATUS_INVALID_VALUE;
  if (generated_count || completed)
    return UDONONNX_STATUS_INVALID_STATE;
  prompt_count = 0;
  *workspace = prompt;
  return UDONONNX_STATUS_SUCCESS;
}

udononnxStatus_t tinyllamaConversationCommitPrompt(uint32_t token_count) {
  uint32_t index;
  udononnxStatus_t status = prompt_replacement_state();
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  if (!token_count)
    return UDONONNX_STATUS_INVALID_VALUE;
  if (token_count > TINYLLAMA_CONTEXT_CAPACITY)
    return UDONONNX_STATUS_CAPACITY_EXCEEDED;
  if (generated_count || completed)
    return UDONONNX_STATUS_INVALID_STATE;
  for (index = 0; index < token_count; ++index)
    if (prompt[index] < 0 || prompt[index] >= TINYLLAMA_LOGIT_WORDS)
      return UDONONNX_STATUS_INVALID_VALUE;
  prompt_count = token_count;
  return UDONONNX_STATUS_SUCCESS;
}

/* Closes the previous reply: a token that was never run is run now, and a
 * reply that did not end with EOS gets one, so the conversation continues
 * after the template's end-of-sequence text exactly as when the model emits
 * EOS itself. */
udononnxStatus_t tinyllamaConversationPrepareTurn(int32_t **workspace,
                                                  uint32_t *capacity) {
  udononnxStatus_t status = prompt_replacement_state();
  uint32_t closing = 0;
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  if (!workspace || !capacity)
    return UDONONNX_STATUS_INVALID_VALUE;
  if (!completed || !generated_count ||
      generated_count > TINYLLAMA_GENERATED_CAPACITY)
    return UDONONNX_STATUS_INVALID_STATE;
  if (last_unrun) {
    prompt[closing++] = generated[generated_count - 1u];
    if (generated[generated_count - 1u] != (int32_t)TINYLLAMA_EOS_TOKEN)
      prompt[closing++] = (int32_t)TINYLLAMA_EOS_TOKEN;
  }
  /* A turn needs at least one token after the closing tokens. */
  if (context_count > TINYLLAMA_CONTEXT_CAPACITY ||
      closing >= TINYLLAMA_CONTEXT_CAPACITY - context_count)
    return UDONONNX_STATUS_CAPACITY_EXCEEDED;
  prompt_count = 0;
  continuing = 0;
  closing_count = closing;
  turn_prepared = 1;
  *workspace = prompt + closing;
  *capacity = TINYLLAMA_CONTEXT_CAPACITY - context_count - closing;
  return UDONONNX_STATUS_SUCCESS;
}

udononnxStatus_t tinyllamaConversationCommitTurn(uint32_t token_count) {
  uint32_t index;
  udononnxStatus_t status = prompt_replacement_state();
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  if (!completed || !turn_prepared)
    return UDONONNX_STATUS_INVALID_STATE;
  if (!token_count)
    return UDONONNX_STATUS_INVALID_VALUE;
  if (token_count > TINYLLAMA_CONTEXT_CAPACITY - context_count - closing_count)
    return UDONONNX_STATUS_CAPACITY_EXCEEDED;
  for (index = closing_count; index < closing_count + token_count; ++index)
    if (prompt[index] < 0 || prompt[index] >= TINYLLAMA_LOGIT_WORDS)
      return UDONONNX_STATUS_INVALID_VALUE;
  prompt_count = closing_count + token_count;
  continuing = 1;
  return UDONONNX_STATUS_SUCCESS;
}

udononnxStatus_t tinyllamaConversationPrompt(const int32_t **tokens,
                                             uint32_t *token_count) {
  if (disposed)
    return UDONONNX_STATUS_DISPOSED;
  if (!initialized)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  if (busy)
    return UDONONNX_STATUS_BUSY;
  if (!tokens || !token_count)
    return UDONONNX_STATUS_INVALID_VALUE;
  *tokens = prompt;
  *token_count = prompt_count;
  return UDONONNX_STATUS_SUCCESS;
}

udononnxStatus_t tinyllamaConversationGenerated(const int32_t **tokens,
                                                uint32_t *token_count) {
  if (disposed)
    return UDONONNX_STATUS_DISPOSED;
  if (!initialized)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  if (busy)
    return UDONONNX_STATUS_BUSY;
  if (!completed)
    return UDONONNX_STATUS_INVALID_STATE;
  if (!tokens || !token_count || generated_count > TINYLLAMA_GENERATED_CAPACITY)
    return UDONONNX_STATUS_INVALID_VALUE;
  *tokens = generated;
  *token_count = generated_count;
  return UDONONNX_STATUS_SUCCESS;
}

udononnxStatus_t
tinyllamaConversationGenerationProgress(const int32_t **tokens,
                                        uint32_t *token_count, uint32_t *serial,
                                        int32_t *is_completed) {
  if (disposed)
    return UDONONNX_STATUS_DISPOSED;
  if (!initialized)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  if (!tokens || !token_count || !serial || !is_completed ||
      generated_count > TINYLLAMA_GENERATED_CAPACITY)
    return UDONONNX_STATUS_INVALID_VALUE;
  *tokens = generated;
  *token_count = generated_count;
  *serial = generation_serial;
  *is_completed = completed && !busy;
  return UDONONNX_STATUS_SUCCESS;
}

/* The position of a run, uploaded without waiting. */
static udononnxStatus_t enqueue_position(uint32_t position) {
  staged_positions[position] = position;
  return clEnqueueWriteBuffer(queue, position_ids, CL_FALSE, 0,
                              sizeof(staged_positions[position]),
                              &staged_positions[position], 0, 0,
                              0) == CL_SUCCESS
             ? UDONONNX_STATUS_SUCCESS
             : UDONONNX_STATUS_OPENCL_ERROR;
}

/* A prompt token runs at its own position. */
static udononnxStatus_t enqueue_prompt_token(int32_t token, uint32_t position) {
  udononnxStatus_t status;
  if (position >= TINYLLAMA_CONTEXT_CAPACITY)
    return UDONONNX_STATUS_INVALID_VALUE;
  staged_tokens[position] = token;
  if (clEnqueueWriteBuffer(queue, input_ids, CL_FALSE, 0,
                           sizeof(staged_tokens[position]),
                           &staged_tokens[position], 0, 0, 0) != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  status = enqueue_position(position);
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  return enqueue_run(position);
}

/* Greedy ArgMax of the logits of the last run into the Device result
 * word. */
static udononnxStatus_t enqueue_select_token(void) {
  if (__vrc_enqueue_udonaten_argmax_reduce_device_args(queue, logits, partials,
                                                       0, TINYLLAMA_LOGIT_WORDS,
                                                       0, 0, 0) != CL_SUCCESS ||
      __vrc_enqueue_udonaten_argmax_finalize_device_args(
          queue, logits, partials, result, 0, TINYLLAMA_LOGIT_WORDS, 1, 0, 0,
          0) != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  return UDONONNX_STATUS_SUCCESS;
}

/* The selected token feeds the next step on the Device: the result word is
 * copied into the low word of input_ids, whose high word every prompt
 * upload left zero. No Host round trip separates two decode steps. */
static udononnxStatus_t enqueue_feed_selected_token(void) {
  return clEnqueueCopyBuffer(queue, result, input_ids, 0, 0, sizeof(uint32_t),
                             0, 0, 0) == CL_SUCCESS
             ? UDONONNX_STATUS_SUCCESS
             : UDONONNX_STATUS_OPENCL_ERROR;
}

/* The Host needs the token only for the EOS check and the generated-token
 * array, so it requests it without waiting and consumes it after the next
 * step is already enqueued. */
static udononnxStatus_t enqueue_read_selected_token(void) {
  return clEnqueueReadBuffer(queue, result, CL_FALSE, 0, sizeof(selected_token),
                             &selected_token, 0, 0,
                             &selected_event) == CL_SUCCESS
             ? UDONONNX_STATUS_SUCCESS
             : UDONONNX_STATUS_OPENCL_ERROR;
}

static udononnxStatus_t wait_selected_token(uint32_t *token) {
  cl_int waited = clWaitForEvents(1, &selected_event);
  cl_int released = clReleaseEvent(selected_event);
  selected_event = 0;
  if (waited != CL_SUCCESS || released != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  *token = selected_token;
  return *token == TINYLLAMA_ARGMAX_FAILURE ? UDONONNX_STATUS_EXECUTION_FAILURE
                                            : UDONONNX_STATUS_SUCCESS;
}

/* One decode step. Unless this is the last requested token, the next model
 * step is enqueued at `position` from the Device-resident token before the
 * Host reads it. When the token turns out to be EOS that speculative step
 * already ran the EOS that ends the reply; its logits are unused, and a next
 * turn continues after it. */
static udononnxStatus_t decode_token(uint32_t position, int last,
                                     uint32_t *token) {
  udononnxStatus_t status = enqueue_select_token(), waited;
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  if (!last) {
    status = enqueue_feed_selected_token();
    if (status == UDONONNX_STATUS_SUCCESS)
      status = enqueue_position(position);
    if (status != UDONONNX_STATUS_SUCCESS)
      return status;
  }
  status = enqueue_read_selected_token();
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  if (!last)
    status = enqueue_run(position);
  waited = wait_selected_token(token);
  return status == UDONONNX_STATUS_SUCCESS ? waited : status;
}

UDONCL_EXPORT udononnxStatus_t generate(uint32_t max_new_tokens) {
  uint32_t index, output_index, token = 0, prompt_tokens, base;
  udononnxStatus_t status = UDONONNX_STATUS_SUCCESS;
  if (disposed)
    return UDONONNX_STATUS_DISPOSED;
  if (!initialized)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  if (busy)
    return UDONONNX_STATUS_BUSY;
  /* After a completed generation only a committed turn continues. */
  if (failed || (completed && !continuing))
    return UDONONNX_STATUS_INVALID_STATE;
  base = continuing ? context_count : 0u;
  prompt_tokens = prompt_count;
  if (!prompt_tokens || !max_new_tokens)
    return UDONONNX_STATUS_INVALID_VALUE;
  /* Every run position stays below the Context Capacity: the prompt runs at
   * [base, base + prompt_tokens) and all but the last new token run after
   * it. A rejected request leaves the conversation unchanged. */
  if (base > TINYLLAMA_CONTEXT_CAPACITY ||
      prompt_tokens > TINYLLAMA_CONTEXT_CAPACITY - base ||
      max_new_tokens > TINYLLAMA_GENERATED_CAPACITY ||
      max_new_tokens - 1u > TINYLLAMA_CONTEXT_CAPACITY - base - prompt_tokens)
    return UDONONNX_STATUS_CAPACITY_EXCEEDED;
  busy = 1;
  completed = 0;
  continuing = turn_prepared = 0;
  generated_count = 0;
  ++generation_serial;
  for (index = 0; index < prompt_tokens; ++index) {
    status = enqueue_prompt_token(prompt[index], base + index);
    if (status != UDONONNX_STATUS_SUCCESS)
      break;
  }
  for (output_index = 0; output_index < max_new_tokens; ++output_index) {
    if (status != UDONONNX_STATUS_SUCCESS)
      break;
    status = decode_token(base + prompt_tokens + output_index,
                          output_index + 1u == max_new_tokens, &token);
    if (status != UDONONNX_STATUS_SUCCESS)
      break;
    generated[output_index] = (int32_t)token;
    generated_count = output_index + 1u;
    if (token == TINYLLAMA_EOS_TOKEN)
      break;
  }
  if (status == UDONONNX_STATUS_SUCCESS) {
    completed = 1;
    /* The last requested token selects but does not run the next step. */
    last_unrun = generated_count == max_new_tokens;
    context_count =
        base + prompt_tokens + generated_count - (uint32_t)last_unrun;
  } else {
    failed = 1;
    completed = 0;
    generated_count = 0;
  }
  busy = 0;
  return status;
}

UDONCL_EXPORT uint32_t generated_token_count(void) { return generated_count; }

UDONCL_EXPORT udononnxStatus_t
read_generated_tokens(udon_int32_array_t destination) {
  uint32_t index, token_count;
  if (disposed)
    return UDONONNX_STATUS_DISPOSED;
  if (!initialized)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  if (busy)
    return UDONONNX_STATUS_BUSY;
  if (!completed)
    return UDONONNX_STATUS_INVALID_STATE;
  token_count = generated_count;
  if (token_count > TINYLLAMA_GENERATED_CAPACITY)
    return UDONONNX_STATUS_EXECUTION_FAILURE;
  if (!destination || udonInt32ArrayLength(destination) < (int32_t)token_count)
    return UDONONNX_STATUS_INVALID_VALUE;
  for (index = 0; index < TINYLLAMA_GENERATED_CAPACITY; ++index) {
    if (index >= token_count)
      break;
    udonInt32ArraySet(destination, (int32_t)index, generated[index]);
  }
  return UDONONNX_STATUS_SUCCESS;
}

UDONCL_EXPORT udononnxStatus_t shutdown(void) {
  udononnxStatus_t status, opencl;
  if (disposed)
    return UDONONNX_STATUS_DISPOSED;
  if (!initialized)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  if (busy)
    return UDONONNX_STATUS_BUSY;
  status = release_session();
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  opencl = release_opencl();
  initialized = 0;
  disposed = 1;
  return opencl;
}
