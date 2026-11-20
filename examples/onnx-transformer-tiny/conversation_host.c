#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdint.h>
#include <udoncl/array.h>
#include <udoncl/export.h>
#include <udoncl/onnx_generated_binding.h>

#include "conversation_host.h"
#define TINYLLAMA_LOGIT_WORDS 32000
#define TINYLLAMA_ARGMAX_PARTIAL_WORDS 512u
#define TINYLLAMA_ARGMAX_FAILURE 0xffffffffu
#define TINYLLAMA_EOS_TOKEN 2u

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

#define TINYLLAMA_INPUT_IDS 0u
#define TINYLLAMA_ATTENTION_MASK 1u
#define TINYLLAMA_POSITION_IDS 2u
#define TINYLLAMA_LOGITS 0u
#define TINYLLAMA_ATTENTION_MASK_BYTES                                         \
  (TINYLLAMA_CONTEXT_CAPACITY * sizeof(int64_t))

static cl_context context;
static cl_command_queue queue;
static cl_mem partials, result;
static int32_t prompt[TINYLLAMA_CONTEXT_CAPACITY];
static int32_t generated[TINYLLAMA_GENERATED_CAPACITY];
static int64_t staged_tokens[TINYLLAMA_CONTEXT_CAPACITY];
static int64_t staged_positions[TINYLLAMA_CONTEXT_CAPACITY];
static const uint32_t zero_word;
static const int64_t one = 1;
/* Destination and event of the one outstanding selected-token read. */
static uint32_t selected_token;
static cl_event selected_event;
static uint32_t prompt_count, generated_count, generation_serial;
static int initialized, disposed, busy, completed, failed;
/* Multi-turn state. The sequence state holds the K/V of positions
 * [0, context_count). After a completed generation the last generated token
 * was never run when the generation reached its limit (last_unrun); a reply
 * that ended with EOS before the limit had its EOS run speculatively. A turn
 * prepared after a completed generation fills prompt[] with the tokens that
 * close the previous reply (closing_count of them) followed by the next turn;
 * once committed (continuing) the next generation feeds them from
 * context_count. */
static uint32_t context_count, closing_count;
static int last_unrun, turn_prepared, continuing;

static void clear_conversation(void) {
  prompt_count = generated_count = context_count = closing_count = 0;
  completed = failed = last_unrun = turn_prepared = continuing = 0;
}

static udononnxStatus_t release_opencl(void) {
  int failed = 0;
  if (selected_event && clReleaseEvent(selected_event) != CL_SUCCESS)
    failed = 1;
  selected_event = 0;
  if (result && clReleaseMemObject(result) != CL_SUCCESS)
    failed = 1;
  result = 0;
  if (partials && clReleaseMemObject(partials) != CL_SUCCESS)
    failed = 1;
  partials = 0;
  if (queue && clReleaseCommandQueue(queue) != CL_SUCCESS)
    failed = 1;
  queue = 0;
  if (context && clReleaseContext(context) != CL_SUCCESS)
    failed = 1;
  context = 0;
  return failed ? UDONONNX_STATUS_OPENCL_ERROR : UDONONNX_STATUS_SUCCESS;
}

static udononnxStatus_t cleanup(udononnxStatus_t status) {
  udononnxStatus_t binding = udon_onnx_binding_dispose();
  udononnxStatus_t opencl = release_opencl();
  initialized = 0;
  if (binding != UDONONNX_STATUS_SUCCESS &&
      binding != UDONONNX_STATUS_NOT_INITIALIZED)
    return binding;
  return opencl == UDONONNX_STATUS_SUCCESS ? status : opencl;
}

/* The generated binding owns every model buffer. Resolve a buffer at each
 * use with a constant binding index instead of caching cl_mem values, so the
 * Host never retains a stale view of binding-owned storage. */
static udononnxStatus_t resolve_input(uint32_t index, size_t expected_length,
                                      cl_mem *memory, size_t *offset) {
  size_t length = 0;
  udononnxStatus_t status =
      udon_onnx_binding_resolve_input(index, memory, offset, &length);
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  return *memory && length == expected_length ? UDONONNX_STATUS_SUCCESS
                                              : UDONONNX_STATUS_TYPE_MISMATCH;
}

static udononnxStatus_t resolve_logits(cl_mem *memory, size_t *offset) {
  size_t length = 0;
  udononnxStatus_t status = udon_onnx_binding_resolve_output(
      TINYLLAMA_LOGITS, memory, offset, &length);
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  return *memory && length == TINYLLAMA_LOGIT_WORDS * sizeof(float) &&
                 *offset % sizeof(float) == 0
             ? UDONONNX_STATUS_SUCCESS
             : UDONONNX_STATUS_TYPE_MISMATCH;
}

static udononnxStatus_t resolve_buffers(void) {
  cl_mem memory = 0;
  size_t offset = 0;
  udononnxStatus_t status =
      resolve_input(TINYLLAMA_INPUT_IDS, sizeof(int64_t), &memory, &offset);
  if (status == UDONONNX_STATUS_SUCCESS)
    status = resolve_input(TINYLLAMA_ATTENTION_MASK,
                           TINYLLAMA_ATTENTION_MASK_BYTES, &memory, &offset);
  if (status == UDONONNX_STATUS_SUCCESS)
    status = resolve_input(TINYLLAMA_POSITION_IDS, sizeof(int64_t), &memory,
                           &offset);
  if (status == UDONONNX_STATUS_SUCCESS)
    status = resolve_logits(&memory, &offset);
  return status;
}

static udononnxStatus_t enqueue_clear_sequence_state(void) {
  cl_mem mask = 0;
  size_t mask_offset = 0;
  udononnxStatus_t status = udon_onnx_binding_enqueue_reset_sequence_state();
  if (status == UDONONNX_STATUS_SUCCESS)
    status = resolve_input(TINYLLAMA_ATTENTION_MASK,
                           TINYLLAMA_ATTENTION_MASK_BYTES, &mask, &mask_offset);
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  if (clEnqueueFillBuffer(queue, mask, &zero_word, sizeof(zero_word),
                          mask_offset, TINYLLAMA_ATTENTION_MASK_BYTES, 0, 0,
                          0) != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  return UDONONNX_STATUS_SUCCESS;
}

UDONCL_EXPORT udononnxStatus_t tinyllama_initialize(void) {
  cl_platform_id platform = 0;
  cl_device_id device = 0;
  cl_int error = CL_SUCCESS;
  udononnxStatus_t status;
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
  status = udon_onnx_binding_create(context, queue);
  if (status == UDONONNX_STATUS_SUCCESS)
    status = udon_onnx_binding_enqueue_initialize();
  if (status == UDONONNX_STATUS_SUCCESS)
    status = udon_onnx_binding_bind_all();
  if (status == UDONONNX_STATUS_SUCCESS)
    status = resolve_buffers();
  if (status != UDONONNX_STATUS_SUCCESS)
    return cleanup(status);
  partials = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            TINYLLAMA_ARGMAX_PARTIAL_WORDS * sizeof(uint32_t),
                            0, &error);
  if (!partials || error != CL_SUCCESS)
    return cleanup(UDONONNX_STATUS_OPENCL_ERROR);
  result =
      clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint32_t), 0, &error);
  if (!result || error != CL_SUCCESS)
    return cleanup(UDONONNX_STATUS_OPENCL_ERROR);
  status = enqueue_clear_sequence_state();
  if (status != UDONONNX_STATUS_SUCCESS)
    return cleanup(status);
  clear_conversation();
  ++generation_serial;
  initialized = 1;
  return UDONONNX_STATUS_SUCCESS;
}

UDONCL_EXPORT udononnxStatus_t tinyllama_reset(void) {
  return tinyllamaConversationReset();
}

udononnxStatus_t tinyllamaConversationReset(void) {
  udononnxStatus_t status;
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
  status = enqueue_clear_sequence_state();
  ++generation_serial;
  if (status == UDONONNX_STATUS_SUCCESS) {
    clear_conversation();
  } else {
    failed = 1;
  }
  return status;
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

UDONCL_EXPORT udononnxStatus_t tinyllama_set_prompt_tokens(
    udon_int32_array_t token_ids, uint32_t token_count) {
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

/* The attention mask and the position of a step depend only on the step
 * index, so the Host patches them on the Device before the token is known. */
static udononnxStatus_t enqueue_step_position(uint32_t position) {
  size_t mask;
  cl_mem attention, positions;
  size_t attention_offset = 0, positions_offset = 0;
  udononnxStatus_t status;
  /* The mask has one word per position; generate() never gets here with a
   * position past the Context Capacity, and this keeps it that way. */
  if (position >= TINYLLAMA_CONTEXT_CAPACITY)
    return UDONONNX_STATUS_CAPACITY_EXCEEDED;
  mask = TINYLLAMA_CONTEXT_CAPACITY - position - 1u;
  status =
      resolve_input(TINYLLAMA_ATTENTION_MASK, TINYLLAMA_ATTENTION_MASK_BYTES,
                    &attention, &attention_offset);
  if (status == UDONONNX_STATUS_SUCCESS)
    status = resolve_input(TINYLLAMA_POSITION_IDS, sizeof(int64_t), &positions,
                           &positions_offset);
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  staged_positions[position] = position;
  if (clEnqueueWriteBuffer(queue, attention, CL_FALSE,
                           attention_offset + mask * sizeof(int64_t),
                           sizeof(one), &one, 0, 0, 0) != CL_SUCCESS ||
      clEnqueueWriteBuffer(queue, positions, CL_FALSE, positions_offset,
                           sizeof(staged_positions[position]),
                           &staged_positions[position], 0, 0, 0) != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  return UDONONNX_STATUS_SUCCESS;
}

/* A prompt step uploads its Host token. */
static udononnxStatus_t enqueue_step(int32_t token, uint32_t position) {
  cl_mem ids;
  size_t ids_offset = 0;
  udononnxStatus_t status =
      resolve_input(TINYLLAMA_INPUT_IDS, sizeof(int64_t), &ids, &ids_offset);
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  staged_tokens[position] = token;
  if (clEnqueueWriteBuffer(queue, ids, CL_FALSE, ids_offset,
                           sizeof(staged_tokens[position]),
                           &staged_tokens[position], 0, 0, 0) != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  status = enqueue_step_position(position);
  return status == UDONONNX_STATUS_SUCCESS ? udon_onnx_binding_enqueue_run()
                                           : status;
}

/* Greedy ArgMax of the current logits into the Device result word. */
static udononnxStatus_t enqueue_select_token(void) {
  cl_mem logits;
  size_t logits_offset = 0;
  cl_int offset;
  udononnxStatus_t status = resolve_logits(&logits, &logits_offset);
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  offset = (cl_int)(logits_offset / sizeof(float));
  if (__vrc_enqueue_udonaten_argmax_reduce_device_args(
          queue, logits, partials, offset, TINYLLAMA_LOGIT_WORDS, 0, 0, 0) !=
          CL_SUCCESS ||
      __vrc_enqueue_udonaten_argmax_finalize_device_args(
          queue, logits, partials, result, offset, TINYLLAMA_LOGIT_WORDS, 1, 0,
          0, 0) != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  return UDONONNX_STATUS_SUCCESS;
}

/* The selected token feeds the next step on the Device: the result word is
 * copied into the low word of input_ids, whose high word every prompt
 * upload left zero. No Host round trip separates two decode steps. */
static udononnxStatus_t enqueue_feed_selected_token(void) {
  cl_mem ids;
  size_t ids_offset = 0;
  udononnxStatus_t status =
      resolve_input(TINYLLAMA_INPUT_IDS, sizeof(int64_t), &ids, &ids_offset);
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  return clEnqueueCopyBuffer(queue, result, ids, 0, ids_offset,
                             sizeof(uint32_t), 0, 0, 0) == CL_SUCCESS
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
 * step is enqueued from the Device-resident token before the Host reads it.
 * When the token turns out to be EOS that speculative step already ran the
 * EOS that ends the reply; its logits are unused, and a next turn continues
 * after it. */
static udononnxStatus_t decode_token(uint32_t position, int last,
                                     uint32_t *token) {
  udononnxStatus_t status = enqueue_select_token(), waited;
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  if (!last) {
    status = udon_onnx_binding_enqueue_advance_sequence_state();
    if (status == UDONONNX_STATUS_SUCCESS)
      status = enqueue_feed_selected_token();
    if (status == UDONONNX_STATUS_SUCCESS)
      status = enqueue_step_position(position);
    if (status != UDONONNX_STATUS_SUCCESS)
      return status;
  }
  status = enqueue_read_selected_token();
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  if (!last)
    status = udon_onnx_binding_enqueue_run();
  waited = wait_selected_token(token);
  return status == UDONONNX_STATUS_SUCCESS ? waited : status;
}

UDONCL_EXPORT udononnxStatus_t tinyllama_generate(uint32_t max_new_tokens) {
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
  /* Each run leaves one pending sequence-state entry that the next step
   * advances first; only the first step of a conversation has none. */
  for (index = 0; index < prompt_tokens; ++index) {
    if (index || base) {
      status = udon_onnx_binding_enqueue_advance_sequence_state();
      if (status != UDONONNX_STATUS_SUCCESS)
        break;
    }
    status = enqueue_step(prompt[index], base + index);
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

UDONCL_EXPORT uint32_t tinyllama_generated_token_count(void) {
  return generated_count;
}

UDONCL_EXPORT udononnxStatus_t
tinyllama_read_generated_tokens(udon_int32_array_t destination) {
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

UDONCL_EXPORT udononnxStatus_t tinyllama_dispose(void) {
  udononnxStatus_t status, opencl;
  if (disposed)
    return UDONONNX_STATUS_DISPOSED;
  if (!initialized)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  if (busy)
    return UDONONNX_STATUS_BUSY;
  status = udon_onnx_binding_dispose();
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  opencl = release_opencl();
  initialized = 0;
  disposed = 1;
  return opencl;
}
