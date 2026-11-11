#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdint.h>
#include <udoncl/array.h>
#include <udoncl/export.h>
#include <udoncl/onnx_generated_binding.h>

#define TINYLLAMA_CONTEXT_CAPACITY 64u
#define TINYLLAMA_GENERATED_CAPACITY 32u
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

static cl_context context;
static cl_command_queue queue;
static cl_mem inputs[3], logits, partials, result;
static size_t input_offsets[3], logits_offset;
static int32_t prompt[TINYLLAMA_CONTEXT_CAPACITY];
static int32_t generated[TINYLLAMA_GENERATED_CAPACITY];
static int64_t staged_tokens[TINYLLAMA_CONTEXT_CAPACITY];
static int64_t staged_positions[TINYLLAMA_CONTEXT_CAPACITY];
static const int64_t zero_attention_mask[TINYLLAMA_CONTEXT_CAPACITY];
static const int64_t one = 1;
static uint32_t prompt_count, generated_count;
static int initialized, disposed, busy, completed, failed;

static udononnxStatus_t release_opencl(void) {
  int failed = 0;
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

static udononnxStatus_t resolve_buffers(void) {
  static const size_t lengths[3] = {8, 512, 8};
  size_t length = 0;
  uint32_t index;
  for (index = 0; index < 3; ++index) {
    udononnxStatus_t status = udon_onnx_binding_resolve_input(
        index, &inputs[index], &input_offsets[index], &length);
    if (status != UDONONNX_STATUS_SUCCESS)
      return status;
    if (!inputs[index] || length != lengths[index])
      return UDONONNX_STATUS_TYPE_MISMATCH;
  }
  {
    udononnxStatus_t status =
        udon_onnx_binding_resolve_output(0, &logits, &logits_offset, &length);
    if (status != UDONONNX_STATUS_SUCCESS)
      return status;
  }
  if (!logits || length != 128000 || logits_offset % sizeof(float))
    return UDONONNX_STATUS_TYPE_MISMATCH;
  return UDONONNX_STATUS_SUCCESS;
}

static udononnxStatus_t enqueue_clear_sequence_state(void) {
  udononnxStatus_t status = udon_onnx_binding_enqueue_reset_sequence_state();
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  if (clEnqueueWriteBuffer(queue, inputs[1], CL_FALSE, input_offsets[1],
                           sizeof(zero_attention_mask), zero_attention_mask, 0,
                           0, 0) != CL_SUCCESS)
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
  prompt_count = generated_count = 0;
  completed = failed = 0;
  initialized = 1;
  return UDONONNX_STATUS_SUCCESS;
}

UDONCL_EXPORT udononnxStatus_t tinyllama_reset(void) {
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
  if (status == UDONONNX_STATUS_SUCCESS) {
    prompt_count = generated_count = 0;
    completed = failed = 0;
  } else {
    failed = 1;
  }
  return status;
}

UDONCL_EXPORT udononnxStatus_t tinyllama_set_prompt_tokens(
    udon_int32_array_t token_ids, uint32_t token_count) {
  uint32_t index;
  if (disposed)
    return UDONONNX_STATUS_DISPOSED;
  if (!initialized)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  if (busy)
    return UDONONNX_STATUS_BUSY;
  if (failed)
    return UDONONNX_STATUS_INVALID_STATE;
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

static udononnxStatus_t enqueue_step(int32_t token, uint32_t position) {
  size_t mask = TINYLLAMA_CONTEXT_CAPACITY - position - 1u;
  staged_tokens[position] = token;
  staged_positions[position] = position;
  if (clEnqueueWriteBuffer(queue, inputs[0], CL_FALSE, input_offsets[0],
                           sizeof(staged_tokens[position]),
                           &staged_tokens[position], 0, 0, 0) != CL_SUCCESS ||
      clEnqueueWriteBuffer(queue, inputs[1], CL_FALSE,
                           input_offsets[1] + mask * sizeof(int64_t),
                           sizeof(one), &one, 0, 0, 0) != CL_SUCCESS ||
      clEnqueueWriteBuffer(queue, inputs[2], CL_FALSE, input_offsets[2],
                           sizeof(staged_positions[position]),
                           &staged_positions[position], 0, 0, 0) != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  return udon_onnx_binding_enqueue_run();
}

static udononnxStatus_t select_token(uint32_t *token) {
  cl_int offset = (cl_int)(logits_offset / sizeof(float));
  if (__vrc_enqueue_udonaten_argmax_reduce_device_args(
          queue, logits, partials, offset, TINYLLAMA_LOGIT_WORDS, 0, 0, 0) !=
          CL_SUCCESS ||
      __vrc_enqueue_udonaten_argmax_finalize_device_args(
          queue, logits, partials, result, offset, TINYLLAMA_LOGIT_WORDS, 1, 0,
          0, 0) != CL_SUCCESS ||
      clEnqueueReadBuffer(queue, result, CL_TRUE, 0, sizeof(*token), token, 0,
                          0, 0) != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  return *token == TINYLLAMA_ARGMAX_FAILURE ? UDONONNX_STATUS_EXECUTION_FAILURE
                                            : UDONONNX_STATUS_SUCCESS;
}

UDONCL_EXPORT udononnxStatus_t tinyllama_generate(uint32_t max_new_tokens) {
  uint32_t index, token = 0;
  udononnxStatus_t status = UDONONNX_STATUS_SUCCESS;
  if (disposed)
    return UDONONNX_STATUS_DISPOSED;
  if (!initialized)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  if (busy)
    return UDONONNX_STATUS_BUSY;
  if (failed || completed)
    return UDONONNX_STATUS_INVALID_STATE;
  if (!prompt_count || !max_new_tokens)
    return UDONONNX_STATUS_INVALID_VALUE;
  if (max_new_tokens > TINYLLAMA_GENERATED_CAPACITY ||
      prompt_count + max_new_tokens - 1u > TINYLLAMA_CONTEXT_CAPACITY)
    return UDONONNX_STATUS_CAPACITY_EXCEEDED;
  busy = 1;
  generated_count = 0;
  for (index = 0; index < prompt_count; ++index) {
    status = enqueue_step(prompt[index], index);
    if (status != UDONONNX_STATUS_SUCCESS)
      break;
    if (index + 1u < prompt_count) {
      status = udon_onnx_binding_enqueue_advance_sequence_state();
      if (status != UDONONNX_STATUS_SUCCESS)
        break;
    }
  }
  while (status == UDONONNX_STATUS_SUCCESS &&
         generated_count < max_new_tokens) {
    status = select_token(&token);
    if (status != UDONONNX_STATUS_SUCCESS)
      break;
    generated[generated_count++] = (int32_t)token;
    if (token == TINYLLAMA_EOS_TOKEN || generated_count == max_new_tokens)
      break;
    status = udon_onnx_binding_enqueue_advance_sequence_state();
    if (status == UDONONNX_STATUS_SUCCESS)
      status =
          enqueue_step((int32_t)token, prompt_count + generated_count - 1u);
  }
  if (status == UDONONNX_STATUS_SUCCESS) {
    completed = 1;
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
  uint32_t index;
  if (disposed)
    return UDONONNX_STATUS_DISPOSED;
  if (!initialized)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  if (busy)
    return UDONONNX_STATUS_BUSY;
  if (!completed)
    return UDONONNX_STATUS_INVALID_STATE;
  if (!destination ||
      udonInt32ArrayLength(destination) < (int32_t)generated_count)
    return UDONONNX_STATUS_INVALID_VALUE;
  for (index = 0; index < generated_count; ++index)
    udonInt32ArraySet(destination, (int32_t)index, generated[index]);
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
