/* Behavior of the UdonONNX session API (udononnxruntime.h) on the native
 * OpenCL runtime. The generated binding of the Host's model is stubbed: its
 * run copies the bound input to the bound output, and its dispose can be made
 * to fail. */
#include <CL/cl.h>
#include <stddef.h>
#include <string.h>
#include <VRC/Runtime/OpenCL/Backend.h>
#include <udononnx_transitional.h>

#define EXPECT(condition)                                                      \
  do {                                                                         \
    if (!(condition))                                                          \
      return __LINE__;                                                         \
  } while (0)

int udononnx_api_conformance(void);

/* Model 3: input "x" and output "y", float32 [2, 2]. Model 4: input "mask"
 * with a dynamic dimension (bound 8) and output "y". */
static const int64_t static_dimensions[2] = {2, 2};
static const int64_t dynamic_dimensions[1] = {-1};
static const int64_t dynamic_bounds[1] = {8};
static const udononnxTensorMetadata_t copy_inputs[1] = {
    {"x", UDONCL_DATA_FLOAT32, 2, static_dimensions, 0}};
static const udononnxTensorMetadata_t copy_outputs[1] = {
    {"y", UDONCL_DATA_FLOAT32, 2, static_dimensions, 0}};
static const udononnxTensorMetadata_t dynamic_inputs[1] = {
    {"mask", UDONCL_DATA_FLOAT32, 1, dynamic_dimensions, dynamic_bounds}};
static const udononnxModel_t copy_model = {"copy",       3, copy_inputs, 1,
                                           copy_outputs, 1};
static const udononnxModel_t dynamic_model = {
    "dynamic", 4, dynamic_inputs, 1, copy_outputs, 1};
static const udononnxModel_t *active_model = &copy_model;

static cl_mem bound_input;
static cl_mem bound_output;
static int creates;
static int disposes;
static udononnxStatus_t dispose_status = UDONONNX_STATUS_SUCCESS;

size_t udononnxGeneratedGetModelCount(void) { return 1; }

udononnxStatus_t udononnxGeneratedGetModel(size_t index,
                                           const udononnxModel_t **model) {
  if (!model || index != 0)
    return UDONONNX_STATUS_INVALID_VALUE;
  *model = active_model;
  return UDONONNX_STATUS_SUCCESS;
}

udononnxStatus_t udononnxGeneratedCreate(uint32_t graph_id,
                                         udononnxTransitionalSession_t *session,
                                         cl_command_queue queue) {
  if (graph_id != active_model->graph_id || session || !queue)
    return UDONONNX_STATUS_INVALID_VALUE;
  ++creates;
  return UDONONNX_STATUS_SUCCESS;
}

udononnxStatus_t udononnxGeneratedBindMemory(uint32_t graph_id, uint32_t output,
                                             uint32_t index, cl_mem memory) {
  if (graph_id != active_model->graph_id || index != 0)
    return UDONONNX_STATUS_INVALID_VALUE;
  if (output)
    bound_output = memory;
  else
    bound_input = memory;
  return UDONONNX_STATUS_SUCCESS;
}

udononnxStatus_t udononnxGeneratedRun(uint32_t graph_id, cl_command_queue queue,
                                      cl_event *completion) {
  if (graph_id != active_model->graph_id || !completion)
    return UDONONNX_STATUS_INVALID_VALUE;
  if (!bound_input || !bound_output)
    return UDONONNX_STATUS_MISSING_BINDING;
  return clEnqueueCopyBuffer(queue, bound_input, bound_output, 0, 0,
                             4 * sizeof(float), 0, 0, completion) == CL_SUCCESS
             ? UDONONNX_STATUS_SUCCESS
             : UDONONNX_STATUS_OPENCL_ERROR;
}

udononnxStatus_t
udononnxGeneratedDispose(uint32_t graph_id,
                         udononnxTransitionalSession_t *session) {
  if (graph_id != active_model->graph_id || session)
    return UDONONNX_STATUS_INVALID_VALUE;
  ++disposes;
  bound_input = 0;
  bound_output = 0;
  return dispose_status;
}

/* The transitional binding is not used here. */
udononnxStatus_t
udononnxGeneratedEnqueueRun(uint32_t graph_id,
                            udononnxTransitionalSession_t *session,
                            cl_command_queue queue, cl_uint wait_count,
                            const cl_event *wait_list, cl_event *completion) {
  (void)graph_id;
  (void)session;
  (void)queue;
  (void)wait_count;
  (void)wait_list;
  (void)completion;
  return UDONONNX_STATUS_NOT_SUPPORTED;
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

int main(void) {
  cl_platform_id platform;
  cl_device_id device;
  cl_context context, other_context;
  cl_command_queue queue, other_queue;
  cl_mem input, output;
  cl_int error;
  udononnxSession_t *session = 0, *second = 0;
  udononnxSession_t *foreign = (udononnxSession_t *)&copy_model;
  const float source[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float result[4] = {0};
  const int64_t dims[2] = {2, 2};
  const int64_t wrong_dims[2] = {2, 3};
  const int64_t larger_buffer[2] = {2, 4};
  const int64_t mask_dims[1] = {8};
  int conformance = udononnx_api_conformance();
  if (conformance)
    return 1000 + conformance;

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
  input = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(source), 0, &error);
  EXPECT(input && error == CL_SUCCESS);
  output =
      clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(result), 0, &error);
  EXPECT(output && error == CL_SUCCESS);
  EXPECT(clEnqueueWriteBuffer(queue, input, CL_TRUE, 0, sizeof(source), source,
                              0, 0, 0) == CL_SUCCESS);

  /* Nothing works before CreateSession. */
  EXPECT(udononnxRun(session) == UDONONNX_STATUS_NOT_INITIALIZED);
  EXPECT(udononnxBindInput(session, "x", input, 2, dims, 0) ==
         UDONONNX_STATUS_NOT_INITIALIZED);
  EXPECT(udononnxReleaseSession(session) == UDONONNX_STATUS_NOT_INITIALIZED);

  /* CreateSession checks its arguments and allocates without a Device
   * command. */
  EXPECT(udononnxCreateSession(context, queue, 0) ==
         UDONONNX_STATUS_INVALID_VALUE);
  EXPECT(udononnxCreateSession(0, queue, &session) ==
             UDONONNX_STATUS_INVALID_VALUE &&
         !session);
  EXPECT(udononnxCreateSession(context, other_queue, &session) ==
             UDONONNX_STATUS_CONTEXT_MISMATCH &&
         !session && creates == 0);
  EXPECT(udononnxCreateSession(context, queue, &session) ==
             UDONONNX_STATUS_SUCCESS &&
         session && creates == 1);

  /* A Host has one live session. */
  second = session;
  EXPECT(udononnxCreateSession(context, queue, &second) ==
             UDONONNX_STATUS_INVALID_STATE &&
         !second && creates == 1);
  EXPECT(udononnxRun(foreign) == UDONONNX_STATUS_INVALID_VALUE);

  /* Every graph input and output is bound before Run. */
  EXPECT(udononnxRun(session) == UDONONNX_STATUS_MISSING_BINDING);

  /* Binds name a graph tensor of the right direction, rank and dimensions. */
  EXPECT(udononnxBindInput(session, "z", input, 2, dims, 0) ==
         UDONONNX_STATUS_INVALID_VALUE);
  EXPECT(udononnxBindInput(session, "y", input, 2, dims, 0) ==
         UDONONNX_STATUS_INVALID_VALUE);
  EXPECT(udononnxBindOutput(session, "x", output, 2, dims, 0) ==
         UDONONNX_STATUS_INVALID_VALUE);
  EXPECT(udononnxBindInput(session, 0, input, 2, dims, 0) ==
         UDONONNX_STATUS_INVALID_VALUE);
  EXPECT(udononnxBindInput(session, "x", 0, 2, dims, 0) ==
         UDONONNX_STATUS_INVALID_VALUE);
  EXPECT(udononnxBindInput(session, "x", input, 1, dims, 0) ==
         UDONONNX_STATUS_INVALID_VALUE);
  EXPECT(udononnxBindInput(session, "x", input, 2, 0, 0) ==
         UDONONNX_STATUS_INVALID_VALUE);
  EXPECT(udononnxBindInput(session, "x", input, 2, wrong_dims, 0) ==
         UDONONNX_STATUS_INVALID_VALUE);
  EXPECT(!bound_input);

  /* A buffer whose extent differs from the tensor is not supported yet. */
  EXPECT(udononnxBindInput(session, "x", input, 2, dims, larger_buffer) ==
         UDONONNX_STATUS_NOT_SUPPORTED);
  EXPECT(!bound_input);

  /* buffer_dims equal to dims is the tensor itself. */
  EXPECT(udononnxBindInput(session, "x", input, 2, dims, dims) ==
             UDONONNX_STATUS_SUCCESS &&
         bound_input == input);
  EXPECT(udononnxRun(session) == UDONONNX_STATUS_MISSING_BINDING);
  EXPECT(udononnxBindOutput(session, "y", output, 2, dims, 0) ==
             UDONONNX_STATUS_SUCCESS &&
         bound_output == output);

  /* Run enqueues the graph on the session queue and returns; a later
   * command on that queue observes the output. */
  EXPECT(udononnxRun(session) == UDONONNX_STATUS_SUCCESS);
  EXPECT(clEnqueueReadBuffer(queue, output, CL_TRUE, 0, sizeof(result), result,
                             0, 0, 0) == CL_SUCCESS);
  EXPECT(memcmp(result, source, sizeof(result)) == 0);

  /* Release forgets the bound buffers, which stay the caller's. */
  EXPECT(udononnxReleaseSession(session) == UDONONNX_STATUS_SUCCESS &&
         disposes == 1 && !bound_input && !bound_output);
  EXPECT(udononnxRun(session) == UDONONNX_STATUS_NOT_INITIALIZED);
  EXPECT(udononnxReleaseSession(session) == UDONONNX_STATUS_NOT_INITIALIZED);
  EXPECT(clEnqueueWriteBuffer(queue, input, CL_TRUE, 0, sizeof(source), source,
                              0, 0, 0) == CL_SUCCESS);

  /* Create after Release gives a new session that binds again. */
  EXPECT(udononnxCreateSession(context, queue, &session) ==
             UDONONNX_STATUS_SUCCESS &&
         creates == 2);
  EXPECT(udononnxRun(session) == UDONONNX_STATUS_MISSING_BINDING);

  /* Release returns the status of releasing the session's buffers and
   * leaves the session released either way. */
  dispose_status = UDONONNX_STATUS_OPENCL_ERROR;
  EXPECT(udononnxReleaseSession(session) == UDONONNX_STATUS_OPENCL_ERROR &&
         disposes == 2);
  dispose_status = UDONONNX_STATUS_SUCCESS;
  EXPECT(udononnxRun(session) == UDONONNX_STATUS_NOT_INITIALIZED);

  /* A dimension the model leaves dynamic is not supported yet. */
  active_model = &dynamic_model;
  EXPECT(udononnxCreateSession(context, queue, &session) ==
         UDONONNX_STATUS_SUCCESS);
  EXPECT(udononnxBindInput(session, "mask", input, 1, mask_dims, 0) ==
         UDONONNX_STATUS_NOT_SUPPORTED);
  EXPECT(!bound_input);
  EXPECT(udononnxReleaseSession(session) == UDONONNX_STATUS_SUCCESS);

  EXPECT(clReleaseMemObject(output) == CL_SUCCESS);
  EXPECT(clReleaseMemObject(input) == CL_SUCCESS);
  EXPECT(clReleaseCommandQueue(other_queue) == CL_SUCCESS);
  EXPECT(clReleaseCommandQueue(queue) == CL_SUCCESS);
  EXPECT(clReleaseContext(other_context) == CL_SUCCESS);
  EXPECT(clReleaseContext(context) == CL_SUCCESS);
  return 0;
}
