/* Behavior of the transitional sequence-state binding runtime
 * (udononnx_transitional.h) on the native OpenCL runtime: the binding-state
 * functions the generated __udononnx_transitional_* binding of the TinyLlama
 * Conversation and Chat Hosts calls. The generated dispatchers of the model
 * are stubbed: create makes one buffer, and the run copies the bound input
 * to the bound output. */
#include <CL/cl.h>
#include <stddef.h>
#include <string.h>
#include <VRC/Runtime/OpenCL/Backend.h>
#include <udoncl/device_memory.h>
#include <udononnx_transitional.h>

#define EXPECT(condition)                                                      \
  do {                                                                         \
    if (!(condition))                                                          \
      return __LINE__;                                                         \
  } while (0)

static cl_mem initialization_buffer;

/* Model 0 (graph 7): input "input" and outputs "output" and "optional_aux",
 * float32 [2, 2]. Model 1 (graph 8): the empty past input of a prefill graph,
 * [1, 4, 0, 64]. */
static const int64_t copy_dimensions[2] = {2, 2};
static const int64_t empty_dimensions[4] = {1, 4, 0, 64};
static const udononnxTensorMetadata_t copy_inputs[1] = {
    {"input", UDONCL_DATA_FLOAT32, 2, copy_dimensions, 0}};
static const udononnxTensorMetadata_t copy_outputs[2] = {
    {"output", UDONCL_DATA_FLOAT32, 2, copy_dimensions, 0},
    {"optional_aux", UDONCL_DATA_FLOAT32, 2, copy_dimensions, 0}};
static const udononnxTensorMetadata_t empty_inputs[1] = {
    {"past_key_values.0.key", UDONCL_DATA_FLOAT32, 4, empty_dimensions, 0}};
static const udononnxModel_t models[2] = {
    {"copy", 7, copy_inputs, 1, copy_outputs, 2},
    {"empty-past", 8, empty_inputs, 1, 0, 0}};

size_t udononnxGeneratedGetModelCount(void) { return 2; }

udononnxStatus_t udononnxGeneratedGetModel(size_t index,
                                           const udononnxModel_t **model) {
  if (!model || index >= 2)
    return UDONONNX_STATUS_INVALID_VALUE;
  *model = &models[index];
  return UDONONNX_STATUS_SUCCESS;
}

size_t udoncl_cl_buffer_limit(void) { return 1024 * 1024; }

int udoncl_cl_gather_profile(const char *name,
                             struct udoncl_cl_gather_profile *profile) {
  (void)name;
  (void)profile;
  return 0;
}

int udoncl_cl_draw_gather(uint32_t kernel, const unsigned char *const *inputs,
                          size_t input_count, size_t bytes,
                          const unsigned char *scalars, size_t scalar_bytes,
                          unsigned char *output) {
  (void)kernel;
  (void)inputs;
  (void)input_count;
  (void)bytes;
  (void)scalars;
  (void)scalar_bytes;
  (void)output;
  return 0;
}

int udoncl_cl_source_matches(const char *source, size_t length) {
  (void)source;
  (void)length;
  return 0;
}

int udoncl_cl_kernel_matches(const char *name) {
  (void)name;
  return 0;
}

int udoncl_cl_draw(uint32_t buffer, uint32_t generation,
                   const unsigned char scalar[4], unsigned char result[4]) {
  (void)buffer;
  (void)generation;
  (void)scalar;
  (void)result;
  return 0;
}

udononnxStatus_t
udononnxGeneratedEnqueueRun(uint32_t graph_id,
                            udononnxTransitionalSession_t *session,
                            cl_command_queue queue, cl_uint wait_count,
                            const cl_event *wait_list, cl_event *completion) {
  cl_mem input, output;
  size_t input_offset, input_bytes, output_offset, output_bytes;
  cl_int error;
  if (graph_id != 7 || !session)
    return UDONONNX_STATUS_INVALID_VALUE;
  if (udononnxSessionResolveInput(session, 0, &input, &input_offset,
                                  &input_bytes) != UDONONNX_STATUS_SUCCESS ||
      udononnxSessionResolveOutput(session, 0, &output, &output_offset,
                                   &output_bytes) != UDONONNX_STATUS_SUCCESS ||
      input_bytes != output_bytes)
    return UDONONNX_STATUS_INVALID_VALUE;
  error = clEnqueueCopyBuffer(queue, input, output, input_offset, output_offset,
                              input_bytes, wait_count, wait_list, completion);
  return error == CL_SUCCESS ? UDONONNX_STATUS_SUCCESS
                             : UDONONNX_STATUS_OPENCL_ERROR;
}

udononnxStatus_t
udononnxGeneratedDispose(uint32_t graph_id,
                         udononnxTransitionalSession_t *session) {
  cl_int error;
  if ((graph_id != 7 && graph_id != 8) || !session)
    return UDONONNX_STATUS_INVALID_VALUE;
  if (!initialization_buffer)
    return UDONONNX_STATUS_SUCCESS;
  error = clReleaseMemObject(initialization_buffer);
  if (error != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  initialization_buffer = 0;
  return UDONONNX_STATUS_SUCCESS;
}

/* Creates the model's own Device buffer (one byte) and enqueues nothing. */
udononnxStatus_t udononnxGeneratedCreate(uint32_t graph_id,
                                         udononnxTransitionalSession_t *session,
                                         cl_command_queue queue) {
  cl_int error;
  (void)queue;
  if ((graph_id != 7 && graph_id != 8) || !session || initialization_buffer)
    return UDONONNX_STATUS_INVALID_VALUE;
  initialization_buffer =
      clCreateBuffer(session->context, CL_MEM_READ_WRITE, 1, 0, &error);
  return initialization_buffer && error == CL_SUCCESS
             ? UDONONNX_STATUS_SUCCESS
             : UDONONNX_STATUS_OPENCL_ERROR;
}

/* The session API is not used here. */
udononnxStatus_t udononnxGeneratedBindMemory(uint32_t graph_id, uint32_t output,
                                             uint32_t index, cl_mem memory,
                                             int32_t extent, int32_t pitch) {
  (void)graph_id;
  (void)output;
  (void)index;
  (void)memory;
  (void)extent;
  (void)pitch;
  return UDONONNX_STATUS_NOT_SUPPORTED;
}

udononnxStatus_t udononnxGeneratedRun(uint32_t graph_id, cl_command_queue queue,
                                      cl_event *completion) {
  (void)graph_id;
  (void)queue;
  (void)completion;
  return UDONONNX_STATUS_NOT_SUPPORTED;
}

/* Writes the binding-table entry the generated bind-all writes for a tensor
 * of `rank` dimensions viewed in `buffer`. */
static int bind(udononnxTensor_t *entry, udonclBufferHandle_t buffer,
                size_t byte_offset, size_t elements, uint32_t rank,
                const int64_t *dimensions) {
  uint32_t axis;
  if (udonclDeviceBufferView(buffer, byte_offset, elements, UDONCL_DATA_FLOAT32,
                             &entry->storage) != UDONCL_STATUS_SUCCESS)
    return 0;
  entry->rank = rank;
  for (axis = 0; axis < UDONONNX_MAX_TENSOR_RANK; ++axis)
    entry->dimensions[axis] = axis < rank ? dimensions[axis] : 0;
  return 1;
}

/* Runs the binding in `state` to completion. */
static udononnxStatus_t run(void *state) {
  cl_event completion = 0;
  udononnxStatus_t status =
      udononnxBindingStateEnqueueRun(state, 0, 0, &completion);
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  if (!completion || clWaitForEvents(1, &completion) != CL_SUCCESS)
    return UDONONNX_STATUS_EXECUTION_FAILURE;
  status = udononnxBindingStateCompleteEvent(state, completion,
                                             UDONONNX_STATUS_SUCCESS);
  return clReleaseEvent(completion) == CL_SUCCESS
             ? status
             : UDONONNX_STATUS_OPENCL_ERROR;
}

int main(void) {
  cl_platform_id platform;
  cl_device_id device;
  cl_context context, other_context;
  cl_command_queue queue, other_queue, binding_queue = 0;
  cl_int error;
  cl_event completion = 0, first = 0;
  cl_mem memory = 0;
  size_t offset = 0, bytes = 0;
  udonclBufferHandle_t input = {0, 0}, output = {0, 0}, read_write = {0, 0};
  udononnxTransitionalSession_t *session = 0;
  _Alignas(8) unsigned char state[UDONONNX_BINDING_STATE_STORAGE_BYTES] = {0};
  udononnxTensor_t inputs[1], outputs[2];
  const float source[4] = {1.0f, -2.0f, 3.5f, 9.0f};
  float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  uint32_t version;

  EXPECT(clGetPlatformIDs(1, &platform, 0) == CL_SUCCESS);
  EXPECT(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, 0) ==
         CL_SUCCESS);
  context = clCreateContext(0, 1, &device, 0, 0, &error);
  EXPECT(context && error == CL_SUCCESS);
  other_context = clCreateContext(0, 1, &device, 0, 0, &error);
  EXPECT(other_context && error == CL_SUCCESS);
  queue = clCreateCommandQueue(context, device, 0, &error);
  EXPECT(queue && error == CL_SUCCESS);
  other_queue = clCreateCommandQueue(other_context, device, 0, &error);
  EXPECT(other_queue && error == CL_SUCCESS);

  /* Create takes a context, a queue of that context and binding tables
   * large enough for the model. */
  EXPECT(udononnxBindingStateBeginStaticBind(state) ==
         UDONONNX_STATUS_NOT_INITIALIZED);
  EXPECT(udononnxBindingStateCreate(state, sizeof(state), 0, queue, 0, inputs,
                                    sizeof(inputs), outputs, sizeof(outputs)) ==
         UDONONNX_STATUS_INVALID_VALUE);
  EXPECT(udononnxBindingStateCreate(state, sizeof(state), context, other_queue,
                                    0, inputs, sizeof(inputs), outputs,
                                    sizeof(outputs)) ==
         UDONONNX_STATUS_CONTEXT_MISMATCH);
  EXPECT(udononnxBindingStateCreate(
             state, sizeof(state), context, queue, 0, inputs, sizeof(inputs),
             outputs, sizeof(outputs[0])) == UDONONNX_STATUS_INVALID_VALUE);
  EXPECT(udononnxBindingStateCreate(
             state, sizeof(state), context, queue, 0, inputs, sizeof(inputs),
             outputs, sizeof(outputs)) == UDONONNX_STATUS_SUCCESS);
  EXPECT(udononnxBindingStateCreate(
             state, sizeof(state), context, queue, 0, inputs, sizeof(inputs),
             outputs, sizeof(outputs)) == UDONONNX_STATUS_INVALID_VALUE);
  /* The session is ready once the generated create has made its own
   * buffers, after the graph inputs and outputs. */
  EXPECT(udononnxBindingStateBeginStaticBind(state) ==
         UDONONNX_STATUS_NOT_INITIALIZED);
  EXPECT(initialization_buffer == 0);
  EXPECT(udononnxBindingStateCreateSession(state) == UDONONNX_STATUS_SUCCESS);
  EXPECT(udononnxBindingStateCreateSession(state) ==
         UDONONNX_STATUS_INVALID_STATE);
  EXPECT(udononnxBindingStateGetCommandQueue(state, &binding_queue) ==
             UDONONNX_STATUS_SUCCESS &&
         binding_queue == queue);
  EXPECT(udononnxBindingStateGetSession(state, &session) ==
             UDONONNX_STATUS_SUCCESS &&
         session && session->model == &models[0]);

  EXPECT(initialization_buffer != 0);
  EXPECT(udononnxBindingStateCompleteEvent(state, (cl_event)state,
                                           UDONONNX_STATUS_SUCCESS) ==
         UDONONNX_STATUS_STALE_HANDLE);

  /* The binding allocates its Device buffers read-only, write-only or
   * read-write. */
  EXPECT(udononnxBindingStateAllocateBuffer(state, 3, sizeof(source), &input) ==
         UDONONNX_STATUS_INVALID_VALUE);
  EXPECT(udononnxBindingStateAllocateBuffer(state, 0, sizeof(source), &input) ==
         UDONONNX_STATUS_SUCCESS);
  EXPECT(udononnxBindingStateAllocateBuffer(
             state, 1, sizeof(result), &output) == UDONONNX_STATUS_SUCCESS);
  EXPECT(udononnxBindingStateAllocateBuffer(
             state, 2, sizeof(source), &read_write) == UDONONNX_STATUS_SUCCESS);
  EXPECT(udonclDeviceBufferGetRaw(read_write, &memory) ==
         UDONCL_STATUS_SUCCESS);
  EXPECT(clEnqueueWriteBuffer(queue, memory, CL_TRUE, 0, sizeof(source), source,
                              0, 0, 0) == CL_SUCCESS);
  EXPECT(clEnqueueReadBuffer(queue, memory, CL_TRUE, 0, sizeof(result), result,
                             0, 0, 0) == CL_SUCCESS);
  EXPECT(memcmp(source, result, sizeof(source)) == 0);

  /* The generated bind-all writes the tables between Begin and Complete,
   * which records its status. A run needs every input and one output. */
  EXPECT(udononnxBindingStateBeginStaticBind(state) == UDONONNX_STATUS_SUCCESS);
  EXPECT(udononnxBindingStateCompleteStaticBind(
             state, UDONONNX_STATUS_INVALID_VALUE) ==
         UDONONNX_STATUS_INVALID_VALUE);
  EXPECT(session->last_status == UDONONNX_STATUS_INVALID_VALUE);
  EXPECT(run(state) == UDONONNX_STATUS_MISSING_BINDING);
  EXPECT(udononnxBindingStateBeginStaticBind(state) == UDONONNX_STATUS_SUCCESS);
  EXPECT(bind(&inputs[0], input, 0, 4, 2, copy_dimensions));
  EXPECT(udononnxBindingStateCompleteStaticBind(
             state, UDONONNX_STATUS_SUCCESS) == UDONONNX_STATUS_SUCCESS);
  EXPECT(run(state) == UDONONNX_STATUS_MISSING_BINDING);
  EXPECT(udononnxBindingStateBeginStaticBind(state) == UDONONNX_STATUS_SUCCESS);
  EXPECT(bind(&outputs[0], output, 0, 4, 2, copy_dimensions));
  EXPECT(udononnxBindingStateCompleteStaticBind(
             state, UDONONNX_STATUS_SUCCESS) == UDONONNX_STATUS_SUCCESS);

  /* The bound tensors resolve to their buffers; an unbound one does not. */
  EXPECT(udononnxBindingStateResolveInput(state, 0, &memory, &offset, &bytes) ==
             UDONONNX_STATUS_SUCCESS &&
         memory && offset == 0 && bytes == sizeof(source));
  EXPECT(
      udononnxBindingStateResolveOutput(state, 1, &memory, &offset, &bytes) ==
      UDONONNX_STATUS_MISSING_BINDING);
  EXPECT(
      udononnxBindingStateResolveOutput(state, 2, &memory, &offset, &bytes) ==
      UDONONNX_STATUS_INVALID_VALUE);

  /* A run copies the input into the output; while it is pending the binding
   * is busy. */
  EXPECT(udononnxBindingStateResolveInput(state, 0, &memory, &offset, &bytes) ==
         UDONONNX_STATUS_SUCCESS);
  EXPECT(clEnqueueWriteBuffer(queue, memory, CL_TRUE, offset, bytes, source, 0,
                              0, 0) == CL_SUCCESS);
  EXPECT(udononnxBindingStateEnqueueRun(state, 0, 0, &first) ==
             UDONONNX_STATUS_SUCCESS &&
         first);
  EXPECT(udononnxBindingStateEnqueueRun(state, 0, 0, &completion) ==
             UDONONNX_STATUS_BUSY &&
         !completion);
  EXPECT(udononnxBindingStateBeginStaticBind(state) == UDONONNX_STATUS_BUSY);
  EXPECT(udononnxBindingStateCompleteStaticBind(
             state, UDONONNX_STATUS_SUCCESS) == UDONONNX_STATUS_SUCCESS);
  EXPECT(udononnxBindingStateDispose(state) == UDONONNX_STATUS_BUSY);
  EXPECT(clWaitForEvents(1, &first) == CL_SUCCESS);
  EXPECT(udononnxBindingStateCompleteEvent(
             state, first, UDONONNX_STATUS_SUCCESS) == UDONONNX_STATUS_SUCCESS);
  EXPECT(clReleaseEvent(first) == CL_SUCCESS);
  memset(result, 0, sizeof(result));
  EXPECT(udononnxBindingStateResolveOutput(state, 0, &memory, &offset,
                                           &bytes) == UDONONNX_STATUS_SUCCESS);
  EXPECT(clEnqueueReadBuffer(queue, memory, CL_TRUE, offset, bytes, result, 0,
                             0, 0) == CL_SUCCESS);
  EXPECT(memcmp(source, result, sizeof(source)) == 0);
  EXPECT(session->last_status == UDONONNX_STATUS_SUCCESS);

  /* The first run validated the table: a run of the unchanged table keeps
   * the binding version and only checks that the bound buffers are alive.
   * A bind-all validates it again under a version no binding had before. */
  EXPECT(run(state) == UDONONNX_STATUS_SUCCESS);
  EXPECT(session->bindings_validated == 1 &&
         session->validated_buffer_epoch == udonclDeviceBufferEpoch());
  version = udononnxSessionBindingVersion(session);
  EXPECT(version != 0 && udononnxSessionBindingVersion(0) == 0);
  EXPECT(run(state) == UDONONNX_STATUS_SUCCESS);
  EXPECT(udononnxSessionBindingVersion(session) == version);
  EXPECT(udononnxBindingStateBeginStaticBind(state) == UDONONNX_STATUS_SUCCESS);
  EXPECT(udononnxBindingStateCompleteStaticBind(
             state, UDONONNX_STATUS_SUCCESS) == UDONONNX_STATUS_SUCCESS);
  EXPECT(session->bindings_validated == 0 &&
         udononnxSessionBindingVersion(session) != version &&
         udononnxSessionBindingVersion(session) != 0);
  EXPECT(run(state) == UDONONNX_STATUS_SUCCESS &&
         session->bindings_validated == 1);

  /* An output may not overlap an input. */
  EXPECT(udononnxBindingStateBeginStaticBind(state) == UDONONNX_STATUS_SUCCESS);
  EXPECT(bind(&outputs[0], input, 0, 4, 2, copy_dimensions));
  EXPECT(udononnxBindingStateCompleteStaticBind(
             state, UDONONNX_STATUS_SUCCESS) == UDONONNX_STATUS_SUCCESS);
  EXPECT(run(state) == UDONONNX_STATUS_INVALID_VALUE);
  EXPECT(udononnxBindingStateBeginStaticBind(state) == UDONONNX_STATUS_SUCCESS);
  EXPECT(bind(&outputs[0], output, 0, 4, 2, copy_dimensions));
  EXPECT(udononnxBindingStateCompleteStaticBind(
             state, UDONONNX_STATUS_SUCCESS) == UDONONNX_STATUS_SUCCESS);
  EXPECT(run(state) == UDONONNX_STATUS_SUCCESS);

  /* A destroyed bound buffer is found by the next run. */
  {
    uint32_t epoch = udonclDeviceBufferEpoch();
    EXPECT(udonclDeviceBufferDestroy(&output) == UDONCL_STATUS_SUCCESS);
    EXPECT(udonclDeviceBufferEpoch() != epoch);
  }
  EXPECT(udononnxBindingStateEnqueueRun(state, 0, 0, &completion) ==
             UDONONNX_STATUS_STALE_HANDLE &&
         !completion);
  EXPECT(session->last_status == UDONONNX_STATUS_STALE_HANDLE);
  EXPECT(udononnxBindingStateDispose(state) == UDONONNX_STATUS_SUCCESS);
  EXPECT(initialization_buffer == 0);
  EXPECT(udononnxBindingStateBeginStaticBind(state) ==
         UDONONNX_STATUS_NOT_INITIALIZED);
  EXPECT(udononnxBindingStateCompleteEvent(state, first,
                                           UDONONNX_STATUS_SUCCESS) ==
         UDONONNX_STATUS_NOT_INITIALIZED);

  /* An empty past input ([1, 4, 0, 64]) is bound to a zero-length view. */
  EXPECT(udononnxBindingStateCreate(state, sizeof(state), context, queue, 1,
                                    inputs, sizeof(inputs), 0,
                                    0) == UDONONNX_STATUS_SUCCESS);
  EXPECT(udononnxBindingStateCreateSession(state) == UDONONNX_STATUS_SUCCESS);
  EXPECT(udononnxBindingStateBeginStaticBind(state) == UDONONNX_STATUS_SUCCESS);
  EXPECT(bind(&inputs[0], input, sizeof(source), 0, 4, empty_dimensions));
  EXPECT(udononnxBindingStateCompleteStaticBind(
             state, UDONONNX_STATUS_SUCCESS) == UDONONNX_STATUS_SUCCESS);
  EXPECT(udononnxBindingStateResolveInput(state, 0, &memory, &offset, &bytes) ==
             UDONONNX_STATUS_SUCCESS &&
         memory && offset == sizeof(source) && bytes == 0);
  EXPECT(udononnxBindingStateDispose(state) == UDONONNX_STATUS_SUCCESS);

  EXPECT(udonclDeviceBufferDestroy(&input) == UDONCL_STATUS_SUCCESS);
  EXPECT(udonclDeviceBufferDestroy(&read_write) == UDONCL_STATUS_SUCCESS);
  EXPECT(clFinish(queue) == CL_SUCCESS);
  EXPECT(clReleaseCommandQueue(queue) == CL_SUCCESS);
  EXPECT(clReleaseCommandQueue(other_queue) == CL_SUCCESS);
  EXPECT(clReleaseContext(other_context) == CL_SUCCESS);
  EXPECT(clReleaseContext(context) == CL_SUCCESS);
  return 0;
}
