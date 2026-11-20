#include <CL/cl.h>
#include <stddef.h>
#include <string.h>
#include <VRC/Runtime/OpenCL/Backend.h>
#include <udoncl/device_memory.h>
#include <udoncl/onnx_binding_abi.h>
#include <udononnxruntime.h>

#define EXPECT(condition)                                                      \
  do {                                                                         \
    if (!(condition))                                                          \
      return __LINE__;                                                         \
  } while (0)

static cl_mem initialization_buffer;

static const int64_t generated_dimensions[1] = {1};
static const udononnxTensorMetadata_t generated_inputs[1] = {
    {"generated_input", UDONCL_DATA_FLOAT32, 1, generated_dimensions, 0}};
static const udononnxTensorMetadata_t generated_outputs[1] = {
    {"generated_output", UDONCL_DATA_FLOAT32, 1, generated_dimensions, 0}};
static const udononnxModel_t generated_model = {
    "generated", 7, generated_inputs, 1, generated_outputs, 1};

size_t udononnxGeneratedGetModelCount(void) { return 1; }

udononnxStatus_t udononnxGeneratedGetModel(size_t index,
                                           const udononnxModel_t **model) {
  if (!model || index != 0)
    return UDONONNX_STATUS_INVALID_VALUE;
  *model = &generated_model;
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

udononnxStatus_t udononnxGeneratedEnqueueInitialize(
    uint32_t graph_id, udononnxSession_t *session, cl_command_queue queue,
    cl_uint wait_count, const cl_event *wait_list, cl_event *completion) {
  static const unsigned char initialized = 1;
  cl_int error;
  if (graph_id != 7 || !session || initialization_buffer)
    return UDONONNX_STATUS_INVALID_VALUE;
  initialization_buffer =
      clCreateBuffer(session->context, CL_MEM_READ_WRITE, 1, 0, &error);
  if (!initialization_buffer || error != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  error = clEnqueueWriteBuffer(queue, initialization_buffer, CL_FALSE, 0, 1,
                               &initialized, wait_count, wait_list, completion);
  return error == CL_SUCCESS ? UDONONNX_STATUS_SUCCESS
                             : UDONONNX_STATUS_OPENCL_ERROR;
}

udononnxStatus_t
udononnxGeneratedEnqueueRun(uint32_t graph_id, udononnxSession_t *session,
                            cl_command_queue queue, cl_uint wait_count,
                            const cl_event *wait_list, cl_event *completion) {
  cl_mem input, output;
  size_t input_offset, input_bytes, output_offset, output_bytes;
  cl_int error;
  if (graph_id != 7 || !session)
    return UDONONNX_STATUS_INVALID_VALUE;
  if (udonclBufferViewResolve(session->input_bindings[0].storage, &input,
                              &input_offset,
                              &input_bytes) != UDONCL_STATUS_SUCCESS ||
      udonclBufferViewResolve(session->output_bindings[0].storage, &output,
                              &output_offset,
                              &output_bytes) != UDONCL_STATUS_SUCCESS ||
      input_bytes != output_bytes)
    return UDONONNX_STATUS_INVALID_VALUE;
  error = clEnqueueCopyBuffer(queue, input, output, input_offset, output_offset,
                              input_bytes, wait_count, wait_list, completion);
  return error == CL_SUCCESS ? UDONONNX_STATUS_SUCCESS
                             : UDONONNX_STATUS_OPENCL_ERROR;
}

udononnxStatus_t udononnxGeneratedDispose(uint32_t graph_id,
                                          udononnxSession_t *session) {
  cl_int error;
  if (graph_id != 7 || !session)
    return UDONONNX_STATUS_INVALID_VALUE;
  if (!initialization_buffer)
    return UDONONNX_STATUS_SUCCESS;
  error = clReleaseMemObject(initialization_buffer);
  if (error != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  initialization_buffer = 0;
  return UDONONNX_STATUS_SUCCESS;
}

int main(void) {
  cl_platform_id platform;
  cl_device_id device;
  cl_context context, other_context;
  cl_command_queue queue, other_queue;
  cl_int error;
  cl_event completion = 0, write = 0, run = 0;
  cl_mem resolved_memory = 0;
  size_t resolved_offset = 0, resolved_bytes = 0;
  udonclBufferHandle_t input_buffer = {0, 0}, output_buffer = {0, 0},
                       foreign_buffer = {0, 0}, binding_input = {0, 0},
                       binding_output = {0, 0}, binding_read_write = {0, 0};
  udonclBufferView_t input_view, output_view, foreign_view;
  udonclBufferView_t empty_view;
  udononnxTensor_t input_tensor = {0}, output_tensor = {0},
                   foreign_tensor = {0}, empty_tensor = {0};
  udononnxTensor_t input_bindings[1], output_bindings[2];
  udononnxSession_t session = {0};
  const udononnxModel_t *selected = 0;
  const udononnxModel_t *constructed_model = 0;
  _Alignas(udononnxTensorMetadata_t) unsigned char
      metadata_storage[2 * UDONONNX_BINDING_METADATA_STORAGE_BYTES];
  _Alignas(udononnxModel_t) unsigned char
      model_storage[UDONONNX_BINDING_MODEL_STORAGE_BYTES];
  _Alignas(udononnxSession_t) unsigned char
      binding_state[UDONONNX_BINDING_STATE_STORAGE_BYTES] = {0};
  _Alignas(udononnxTensor_t) unsigned char
      binding_inputs[UDONONNX_BINDING_TENSOR_STORAGE_BYTES] = {0};
  _Alignas(udononnxTensor_t) unsigned char
      binding_outputs[UDONONNX_BINDING_TENSOR_STORAGE_BYTES] = {0};
  cl_context binding_context = 0;
  cl_command_queue binding_queue = 0;
  udononnxSession_t *binding_session = 0;
  int64_t dimensions[2] = {2, 2};
  int64_t empty_dimensions[4] = {1, 4, 0, 64};
  int32_t binding_dimensions[2] = {2, 2};
  int32_t generated_binding_dimensions[1] = {1};
  const udononnxTensorMetadata_t inputs[1] = {
      {"input", UDONCL_DATA_FLOAT32, 2, dimensions, 0}};
  const udononnxTensorMetadata_t outputs[2] = {
      {"output", UDONCL_DATA_FLOAT32, 2, dimensions, 0},
      {"optional_aux", UDONCL_DATA_FLOAT32, 2, dimensions, 0}};
  const udononnxModel_t models[1] = {
      {"synthetic-copy", 7, inputs, 1, outputs, 2}};
  const udononnxTensorMetadata_t empty_inputs[1] = {
      {"past_key_values.0.key", UDONCL_DATA_FLOAT32, 4, empty_dimensions, 0}};
  const udononnxModel_t empty_model = {"empty-past", 7, empty_inputs, 1, 0, 0};
  const float source[4] = {1.0f, -2.0f, 3.5f, 9.0f};
  float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  EXPECT(strcmp(udononnxStatusString(UDONONNX_STATUS_INVALID_STATE),
                "invalid state") == 0);
  EXPECT(strcmp(udononnxStatusString(UDONONNX_STATUS_CAPACITY_EXCEEDED),
                "capacity exceeded") == 0);

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
  EXPECT(udononnxSessionBindInput(&session, 0, input_tensor) ==
         UDONONNX_STATUS_NOT_INITIALIZED);
  EXPECT(udononnxGetModelCount() == 1);
  EXPECT(udononnxGetModel(0, &selected) == UDONONNX_STATUS_SUCCESS &&
         selected == &generated_model);
  EXPECT(udononnxFindGeneratedModel("generated", &selected) ==
             UDONONNX_STATUS_SUCCESS &&
         selected == &generated_model);
  EXPECT(udononnxBindingInitializeTensorMetadata(
             metadata_storage, sizeof(metadata_storage), 0, "input",
             UDONCL_DATA_FLOAT32, 2, dimensions, 0) == UDONONNX_STATUS_SUCCESS);
  EXPECT(udononnxBindingInitializeTensorMetadata(
             metadata_storage, sizeof(metadata_storage), 1, "output",
             UDONCL_DATA_FLOAT32, 2, dimensions, 0) == UDONONNX_STATUS_SUCCESS);
  EXPECT(udononnxBindingInitializeModel(
             model_storage, sizeof(model_storage), "constructed", 9,
             metadata_storage, 1,
             metadata_storage + sizeof(udononnxTensorMetadata_t), 1,
             &constructed_model) == UDONONNX_STATUS_SUCCESS);
  EXPECT(constructed_model && constructed_model->graph_id == 9 &&
         strcmp(constructed_model->inputs[0].name, "input") == 0 &&
         strcmp(constructed_model->outputs[0].name, "output") == 0);
  EXPECT(udononnxFindModel(models, 1, "synthetic-copy", &selected) ==
             UDONONNX_STATUS_SUCCESS &&
         selected == &models[0]);
  EXPECT(udononnxCreateSession(&session, selected, context, input_bindings,
                               output_bindings) == UDONONNX_STATUS_SUCCESS);
  EXPECT(udononnxSessionGetInputCount(&session) == 1);
  EXPECT(udononnxSessionGetOutputCount(&session) == 2);
  EXPECT(udononnxSessionFindInput(&session, "input") == 0);
  EXPECT(udononnxSessionFindOutput(&session, "output") == 0);
  EXPECT(udononnxSessionFindOutput(&session, "optional_aux") == 1);
  EXPECT(udononnxEnqueueInitialize(&session, queue, 0, 0, &completion) ==
             UDONONNX_STATUS_SUCCESS &&
         completion);
  EXPECT(udononnxSessionDispose(&session) == UDONONNX_STATUS_BUSY);
  EXPECT(clWaitForEvents(1, &completion) == CL_SUCCESS);
  EXPECT(udononnxSessionCompleteEvent(&session, completion,
                                      UDONONNX_STATUS_SUCCESS) ==
         UDONONNX_STATUS_SUCCESS);
  EXPECT(clReleaseEvent(completion) == CL_SUCCESS);

  EXPECT(udonclDeviceBufferCreate(context, CL_MEM_READ_ONLY, sizeof(source),
                                  &input_buffer) == UDONCL_STATUS_SUCCESS);
  EXPECT(udonclDeviceBufferCreate(context, CL_MEM_WRITE_ONLY, sizeof(result),
                                  &output_buffer) == UDONCL_STATUS_SUCCESS);
  EXPECT(udonclDeviceBufferCreate(other_context, CL_MEM_READ_ONLY,
                                  sizeof(source),
                                  &foreign_buffer) == UDONCL_STATUS_SUCCESS);
  EXPECT(udonclDeviceBufferView(input_buffer, 0, 4, UDONCL_DATA_FLOAT32,
                                &input_view) == UDONCL_STATUS_SUCCESS);
  EXPECT(udonclDeviceBufferView(output_buffer, 0, 4, UDONCL_DATA_FLOAT32,
                                &output_view) == UDONCL_STATUS_SUCCESS);
  EXPECT(udonclDeviceBufferView(foreign_buffer, 0, 4, UDONCL_DATA_FLOAT32,
                                &foreign_view) == UDONCL_STATUS_SUCCESS);
  EXPECT(udonclDeviceBufferView(input_buffer, sizeof(source), 0,
                                UDONCL_DATA_FLOAT32,
                                &empty_view) == UDONCL_STATUS_SUCCESS);
  EXPECT(udonclBufferViewResolve(empty_view, &resolved_memory, &resolved_offset,
                                 &resolved_bytes) == UDONCL_STATUS_SUCCESS &&
         resolved_memory && resolved_offset == sizeof(source) &&
         resolved_bytes == 0);
  EXPECT(udononnxCreateTensor(input_view, 2, dimensions, &input_tensor) ==
         UDONONNX_STATUS_SUCCESS);
  EXPECT(udononnxCreateTensor(output_view, 2, dimensions, &output_tensor) ==
         UDONONNX_STATUS_SUCCESS);
  EXPECT(udononnxCreateTensor(foreign_view, 2, dimensions, &foreign_tensor) ==
         UDONONNX_STATUS_SUCCESS);
  EXPECT(udononnxCreateTensor(empty_view, 4, empty_dimensions, &empty_tensor) ==
         UDONONNX_STATUS_SUCCESS);
  EXPECT(udononnxSessionBindInput(&session, 0, foreign_tensor) ==
         UDONONNX_STATUS_CONTEXT_MISMATCH);
  EXPECT(udononnxBindingBindInput(&session, 0, input_buffer.slot,
                                  input_buffer.generation, 0, sizeof(source),
                                  UDONCL_DATA_FLOAT32, 2, binding_dimensions) ==
         UDONONNX_STATUS_SUCCESS);
  EXPECT(udononnxEnqueueRun(&session, queue, 0, 0, &run) ==
         UDONONNX_STATUS_MISSING_BINDING);
  EXPECT(!run);
  EXPECT(udononnxSessionBindOutput(&session, 0, input_tensor) ==
         UDONONNX_STATUS_SUCCESS);
  EXPECT(udononnxEnqueueRun(&session, queue, 0, 0, &run) ==
         UDONONNX_STATUS_INVALID_VALUE);
  EXPECT(udononnxBindingBindOutput(
             &session, 0, output_buffer.slot, output_buffer.generation, 0,
             sizeof(result), UDONCL_DATA_FLOAT32, 2,
             binding_dimensions) == UDONONNX_STATUS_SUCCESS);
  EXPECT(udononnxSessionResolveInput(&session, 0, &resolved_memory,
                                     &resolved_offset, &resolved_bytes) ==
             UDONONNX_STATUS_SUCCESS &&
         resolved_memory && resolved_offset == 0 &&
         resolved_bytes == sizeof(source));
  EXPECT(udononnxSessionResolveOutput(&session, 1, &resolved_memory,
                                      &resolved_offset, &resolved_bytes) ==
         UDONONNX_STATUS_MISSING_BINDING);

  {
    cl_mem input_memory;
    size_t offset, bytes;
    EXPECT(udonclBufferViewResolve(input_view, &input_memory, &offset,
                                   &bytes) == UDONCL_STATUS_SUCCESS);
    EXPECT(clEnqueueWriteBuffer(queue, input_memory, CL_FALSE, offset, bytes,
                                source, 0, 0, &write) == CL_SUCCESS);
  }
  EXPECT(udononnxEnqueueRun(&session, queue, 1, &write, &run) ==
             UDONONNX_STATUS_SUCCESS &&
         run);
  EXPECT(clReleaseEvent(write) == CL_SUCCESS);
  EXPECT(udononnxSessionBindInput(&session, 0, input_tensor) ==
         UDONONNX_STATUS_BUSY);
  EXPECT(udononnxEnqueueRun(&session, queue, 0, 0, &completion) ==
         UDONONNX_STATUS_BUSY);
  EXPECT(udononnxSessionDispose(&session) == UDONONNX_STATUS_BUSY);
  EXPECT(clWaitForEvents(1, &run) == CL_SUCCESS);
  EXPECT(udononnxSessionCompleteEvent(&session, run, UDONONNX_STATUS_SUCCESS) ==
         UDONONNX_STATUS_SUCCESS);
  EXPECT(clReleaseEvent(run) == CL_SUCCESS);
  {
    cl_mem output_memory;
    size_t offset, bytes;
    EXPECT(udonclBufferViewResolve(output_view, &output_memory, &offset,
                                   &bytes) == UDONCL_STATUS_SUCCESS);
    EXPECT(clEnqueueReadBuffer(queue, output_memory, CL_TRUE, offset, bytes,
                               result, 0, 0, 0) == CL_SUCCESS);
  }
  EXPECT(memcmp(source, result, sizeof(source)) == 0);
  EXPECT(udononnxSessionGetLastStatus(&session) == UDONONNX_STATUS_SUCCESS);
  EXPECT(strcmp(udononnxSessionGetLastError(&session), "success") == 0);

  /* The first run validated the binding table. A run of the unchanged table
   * only checks that the bound buffers are alive (nothing while no handle was
   * destroyed); a bind validates the table again, and a destroyed buffer is
   * still reported. The binding version changes with every bind or
   * invalidation and never repeats. */
  EXPECT(session.bindings_validated == 1);
  {
    uint32_t version = udononnxSessionBindingVersion(&session);
    EXPECT(version != 0 &&
           session.validated_buffer_epoch == udonclDeviceBufferEpoch());
    EXPECT(udononnxEnqueueRun(&session, queue, 0, 0, &run) ==
               UDONONNX_STATUS_SUCCESS &&
           run && session.bindings_validated == 1);
    EXPECT(udononnxSessionBindingVersion(&session) == version);
    EXPECT(udononnxSessionBindingVersion(0) == 0);
  }
  EXPECT(clWaitForEvents(1, &run) == CL_SUCCESS);
  EXPECT(udononnxSessionCompleteEvent(&session, run, UDONONNX_STATUS_SUCCESS) ==
         UDONONNX_STATUS_SUCCESS);
  EXPECT(clReleaseEvent(run) == CL_SUCCESS);
  {
    uint32_t before = udononnxSessionBindingVersion(&session);
    EXPECT(udononnxSessionBindOutput(&session, 0, input_tensor) ==
               UDONONNX_STATUS_SUCCESS &&
           session.bindings_validated == 0);
    EXPECT(udononnxSessionBindingVersion(&session) != before &&
           udononnxSessionBindingVersion(&session) != 0);
  }
  EXPECT(udononnxEnqueueRun(&session, queue, 0, 0, &run) ==
             UDONONNX_STATUS_INVALID_VALUE &&
         session.bindings_validated == 0);
  EXPECT(udononnxSessionBindOutput(&session, 0, output_tensor) ==
         UDONONNX_STATUS_SUCCESS);
  EXPECT(udononnxEnqueueRun(&session, queue, 0, 0, &run) ==
             UDONONNX_STATUS_SUCCESS &&
         run && session.bindings_validated == 1);
  EXPECT(clWaitForEvents(1, &run) == CL_SUCCESS);
  EXPECT(udononnxSessionCompleteEvent(&session, run, UDONONNX_STATUS_SUCCESS) ==
         UDONONNX_STATUS_SUCCESS);
  EXPECT(clReleaseEvent(run) == CL_SUCCESS);
  {
    uint32_t before = udononnxSessionBindingVersion(&session);
    udononnxSessionInvalidateBindings(&session);
    EXPECT(session.bindings_validated == 0 &&
           udononnxSessionBindingVersion(&session) != before);
  }
  EXPECT(udononnxEnqueueRun(&session, queue, 0, 0, &run) ==
             UDONONNX_STATUS_SUCCESS &&
         run && session.bindings_validated == 1);
  EXPECT(clWaitForEvents(1, &run) == CL_SUCCESS);
  EXPECT(udononnxSessionCompleteEvent(&session, run, UDONONNX_STATUS_SUCCESS) ==
         UDONONNX_STATUS_SUCCESS);
  EXPECT(clReleaseEvent(run) == CL_SUCCESS);
  {
    uint32_t epoch = udonclDeviceBufferEpoch();
    EXPECT(udonclDeviceBufferDestroy(&output_buffer) == UDONCL_STATUS_SUCCESS);
    EXPECT(udonclDeviceBufferEpoch() != epoch);
  }
  EXPECT(udononnxEnqueueRun(&session, queue, 0, 0, &completion) ==
             UDONONNX_STATUS_STALE_HANDLE &&
         !completion);
  EXPECT(udononnxSessionGetLastStatus(&session) ==
         UDONONNX_STATUS_STALE_HANDLE);
  EXPECT(udononnxSessionDispose(&session) == UDONONNX_STATUS_SUCCESS);
  EXPECT(udononnxSessionCompleteEvent(&session, run, UDONONNX_STATUS_SUCCESS) ==
         UDONONNX_STATUS_STALE_HANDLE);

  EXPECT(udononnxCreateSession(&session, &empty_model, context, input_bindings,
                               0) == UDONONNX_STATUS_SUCCESS);
  EXPECT(udononnxEnqueueInitialize(&session, queue, 0, 0, &completion) ==
             UDONONNX_STATUS_SUCCESS &&
         completion);
  EXPECT(clWaitForEvents(1, &completion) == CL_SUCCESS);
  EXPECT(udononnxSessionCompleteEvent(&session, completion,
                                      UDONONNX_STATUS_SUCCESS) ==
         UDONONNX_STATUS_SUCCESS);
  EXPECT(clReleaseEvent(completion) == CL_SUCCESS);
  EXPECT(udononnxSessionBindInput(&session, 0, empty_tensor) ==
         UDONONNX_STATUS_SUCCESS);
  EXPECT(udononnxSessionResolveInput(&session, 0, &resolved_memory,
                                     &resolved_offset, &resolved_bytes) ==
             UDONONNX_STATUS_SUCCESS &&
         resolved_memory && resolved_offset == sizeof(source) &&
         resolved_bytes == 0);
  EXPECT(udononnxSessionDispose(&session) == UDONONNX_STATUS_SUCCESS);

  EXPECT(udonclDeviceBufferDestroy(&input_buffer) == UDONCL_STATUS_SUCCESS);
  EXPECT(udonclDeviceBufferDestroy(&foreign_buffer) == UDONCL_STATUS_SUCCESS);
  EXPECT(initialization_buffer == 0);
  EXPECT(udononnxBindingStateCreate(
             binding_state, sizeof(binding_state), 0, queue, 0, binding_inputs,
             sizeof(binding_inputs), binding_outputs,
             sizeof(binding_outputs)) == UDONONNX_STATUS_INVALID_VALUE);
  EXPECT(udononnxBindingStateCreate(
             binding_state, sizeof(binding_state), context, other_queue, 0,
             binding_inputs, sizeof(binding_inputs), binding_outputs,
             sizeof(binding_outputs)) == UDONONNX_STATUS_CONTEXT_MISMATCH);
  EXPECT(udononnxBindingStateCreate(
             binding_state, sizeof(binding_state), context, queue, 0,
             binding_inputs, sizeof(binding_inputs), binding_outputs,
             sizeof(binding_outputs)) == UDONONNX_STATUS_SUCCESS);
  EXPECT(udononnxBindingStateBeginStaticBind(binding_state) ==
         UDONONNX_STATUS_NOT_INITIALIZED);
  EXPECT(udononnxBindingStateGetContext(binding_state, &binding_context) ==
             UDONONNX_STATUS_SUCCESS &&
         binding_context == context);
  EXPECT(udononnxBindingStateGetCommandQueue(binding_state, &binding_queue) ==
             UDONONNX_STATUS_SUCCESS &&
         binding_queue == queue);
  completion = 0;
  EXPECT(udononnxBindingStateEnqueueInitialize(
             binding_state, 0, 0, &completion) == UDONONNX_STATUS_SUCCESS &&
         completion);
  EXPECT(clWaitForEvents(1, &completion) == CL_SUCCESS);
  EXPECT(udononnxBindingStateCompleteEvent(binding_state, completion,
                                           UDONONNX_STATUS_SUCCESS) ==
         UDONONNX_STATUS_SUCCESS);
  EXPECT(clReleaseEvent(completion) == CL_SUCCESS);
  EXPECT(udononnxBindingStateBeginStaticBind(binding_state) ==
         UDONONNX_STATUS_SUCCESS);
  EXPECT(udononnxBindingStateCompleteStaticBind(
             binding_state, UDONONNX_STATUS_INVALID_VALUE) ==
         UDONONNX_STATUS_INVALID_VALUE);
  EXPECT(udononnxBindingStateGetSession(binding_state, &binding_session) ==
             UDONONNX_STATUS_SUCCESS &&
         binding_session &&
         binding_session->last_status == UDONONNX_STATUS_INVALID_VALUE);
  EXPECT(udononnxBindingStateCompleteStaticBind(binding_state,
                                                UDONONNX_STATUS_SUCCESS) ==
         UDONONNX_STATUS_SUCCESS);
  EXPECT(udononnxBindingStateAllocateBuffer(binding_state, 3, sizeof(source),
                                            &binding_input) ==
         UDONONNX_STATUS_INVALID_VALUE);
  EXPECT(udononnxBindingStateAllocateBuffer(binding_state, 0, sizeof(float),
                                            &binding_input) ==
         UDONONNX_STATUS_SUCCESS);
  EXPECT(udononnxBindingStateAllocateBuffer(binding_state, 1, sizeof(float),
                                            &binding_output) ==
         UDONONNX_STATUS_SUCCESS);
  EXPECT(udononnxBindingStateAllocateBuffer(binding_state, 2, sizeof(source),
                                            &binding_read_write) ==
         UDONONNX_STATUS_SUCCESS);
  EXPECT(udonclDeviceBufferGetRaw(binding_read_write, &resolved_memory) ==
         UDONCL_STATUS_SUCCESS);
  EXPECT(clEnqueueWriteBuffer(queue, resolved_memory, CL_TRUE, 0,
                              sizeof(source), source, 0, 0, 0) == CL_SUCCESS);
  memset(result, 0, sizeof(result));
  EXPECT(clEnqueueReadBuffer(queue, resolved_memory, CL_TRUE, 0, sizeof(source),
                             result, 0, 0, 0) == CL_SUCCESS);
  EXPECT(memcmp(source, result, sizeof(source)) == 0);
  EXPECT(udonclDeviceBufferGetSize(binding_input, &resolved_bytes) ==
             UDONCL_STATUS_SUCCESS &&
         resolved_bytes == sizeof(float));
  EXPECT(udononnxBindingStateBindTensor(
             binding_state, 0, 0, binding_input.slot, binding_input.generation,
             0, sizeof(float), UDONCL_DATA_FLOAT32, 1,
             generated_binding_dimensions) == UDONONNX_STATUS_SUCCESS);
  EXPECT(udononnxBindingStateBindTensor(
             binding_state, 1, 0, binding_output.slot,
             binding_output.generation, 0, sizeof(float), UDONCL_DATA_FLOAT32,
             1, generated_binding_dimensions) == UDONONNX_STATUS_SUCCESS);
  EXPECT(udononnxBindingStateResolveInput(binding_state, 0, &resolved_memory,
                                          &resolved_offset, &resolved_bytes) ==
             UDONONNX_STATUS_SUCCESS &&
         resolved_memory && resolved_offset == 0 &&
         resolved_bytes == sizeof(float));
  EXPECT(udononnxBindingStateResolveOutput(binding_state, 0, &resolved_memory,
                                           &resolved_offset, &resolved_bytes) ==
             UDONONNX_STATUS_SUCCESS &&
         resolved_memory && resolved_offset == 0 &&
         resolved_bytes == sizeof(float));
  EXPECT(udononnxBindingStateBindTensor(binding_state, 2, 0, 0, 0, 0, 0,
                                        UDONCL_DATA_FLOAT32, 0,
                                        0) == UDONONNX_STATUS_INVALID_VALUE);
  EXPECT(udononnxBindingStateGetSession(binding_state, &binding_session) ==
             UDONONNX_STATUS_SUCCESS &&
         binding_session);
  EXPECT(udonclDeviceBufferDestroy(&binding_input) == UDONCL_STATUS_SUCCESS);
  EXPECT(udonclDeviceBufferDestroy(&binding_output) == UDONCL_STATUS_SUCCESS);
  EXPECT(udonclDeviceBufferDestroy(&binding_read_write) ==
         UDONCL_STATUS_SUCCESS);
  EXPECT(udononnxBindingStateDispose(binding_state) == UDONONNX_STATUS_SUCCESS);
  EXPECT(udononnxBindingStateBeginStaticBind(binding_state) ==
         UDONONNX_STATUS_NOT_INITIALIZED);
  EXPECT(clFinish(queue) == CL_SUCCESS);
  EXPECT(clReleaseCommandQueue(queue) == CL_SUCCESS);
  EXPECT(clReleaseCommandQueue(other_queue) == CL_SUCCESS);
  EXPECT(clReleaseContext(other_context) == CL_SUCCESS);
  EXPECT(clReleaseContext(context) == CL_SUCCESS);
  return 0;
}
