#include <udoncl/onnx_binding_abi.h>
#include <udononnxruntime.h>

#define UDONONNX_BINDING_STATE_MAGIC 0x554f4e58u

struct udononnxBindingState {
  uint32_t magic;
  cl_context context;
  cl_command_queue queue;
  udononnxSession_t session;
};

static struct udononnxBindingState *
udononnxBindingStateGet(void *state_storage) {
  struct udononnxBindingState *state =
      (struct udononnxBindingState *)state_storage;
  return state && state->magic == UDONONNX_BINDING_STATE_MAGIC ? state : 0;
}

udononnxStatus_t udononnxBindingInitializeTensorMetadata(
    void *metadata_storage, uint32_t metadata_storage_bytes, uint32_t index,
    const char *name, udonclDataType_t data_type, uint32_t rank,
    const int64_t *dimensions, const int64_t *dimension_bounds) {
  udononnxTensorMetadata_t *metadata;
  size_t required;
  if (!metadata_storage || !name || !name[0] ||
      rank > UDONONNX_MAX_TENSOR_RANK || (rank && !dimensions) ||
      !udonclDataTypeSize(data_type) ||
      ((uint64_t)index + 1u) * sizeof(udononnxTensorMetadata_t) > SIZE_MAX)
    return UDONONNX_STATUS_INVALID_VALUE;
  required = ((size_t)index + 1u) * sizeof(udononnxTensorMetadata_t);
  if (required > metadata_storage_bytes)
    return UDONONNX_STATUS_INVALID_VALUE;
  metadata = (udononnxTensorMetadata_t *)metadata_storage + index;
  metadata->name = name;
  metadata->data_type = data_type;
  metadata->rank = rank;
  metadata->dimensions = dimensions;
  metadata->dimension_bounds = dimension_bounds;
  return UDONONNX_STATUS_SUCCESS;
}

udononnxStatus_t udononnxBindingInitializeModel(
    void *model_storage, uint32_t model_storage_bytes, const char *name,
    uint32_t graph_id, void *input_metadata, uint32_t input_count,
    void *output_metadata, uint32_t output_count,
    const udononnxModel_t **model) {
  udononnxModel_t *value;
  if (!model_storage || model_storage_bytes < sizeof(udononnxModel_t) ||
      !name || !name[0] || !model || (input_count && !input_metadata) ||
      (output_count && !output_metadata))
    return UDONONNX_STATUS_INVALID_VALUE;
  value = (udononnxModel_t *)model_storage;
  value->name = name;
  value->graph_id = graph_id;
  value->inputs = (const udononnxTensorMetadata_t *)input_metadata;
  value->input_count = input_count;
  value->outputs = (const udononnxTensorMetadata_t *)output_metadata;
  value->output_count = output_count;
  *model = value;
  return UDONONNX_STATUS_SUCCESS;
}

udononnxStatus_t udononnxBindingCreateTensor(
    uint32_t buffer_slot, uint32_t buffer_generation, uint32_t byte_offset,
    uint32_t byte_length, udonclDataType_t data_type, uint32_t rank,
    const int32_t *dimensions, udononnxTensor_t *tensor) {
  uint32_t axis;
  int64_t widened_dimensions[UDONONNX_MAX_TENSOR_RANK];
  udonclBufferView_t view;
  if (!tensor || rank > UDONONNX_MAX_TENSOR_RANK || (rank && !dimensions))
    return UDONONNX_STATUS_INVALID_VALUE;
  view.buffer.slot = buffer_slot;
  view.buffer.generation = buffer_generation;
  view.byte_offset = byte_offset;
  view.byte_length = byte_length;
  view.data_type = data_type;
  for (axis = 0; axis < rank; ++axis)
    widened_dimensions[axis] = dimensions[axis];
  return udononnxCreateTensor(view, rank, widened_dimensions, tensor);
}

static udononnxStatus_t
udononnxBindingBind(udononnxSession_t *session, uint32_t index,
                    uint32_t buffer_slot, uint32_t buffer_generation,
                    uint32_t byte_offset, uint32_t byte_length,
                    udonclDataType_t data_type, uint32_t rank,
                    const int32_t *dimensions, int output) {
  udononnxTensor_t tensor;
  udononnxStatus_t status = udononnxBindingCreateTensor(
      buffer_slot, buffer_generation, byte_offset, byte_length, data_type, rank,
      dimensions, &tensor);
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  return output ? udononnxSessionBindOutput(session, index, tensor)
                : udononnxSessionBindInput(session, index, tensor);
}

udononnxStatus_t udononnxBindingBindInput(
    udononnxSession_t *session, uint32_t input_index, uint32_t buffer_slot,
    uint32_t buffer_generation, uint32_t byte_offset, uint32_t byte_length,
    udonclDataType_t data_type, uint32_t rank, const int32_t *dimensions) {
  return udononnxBindingBind(session, input_index, buffer_slot,
                             buffer_generation, byte_offset, byte_length,
                             data_type, rank, dimensions, 0);
}

udononnxStatus_t udononnxBindingBindOutput(
    udononnxSession_t *session, uint32_t output_index, uint32_t buffer_slot,
    uint32_t buffer_generation, uint32_t byte_offset, uint32_t byte_length,
    udonclDataType_t data_type, uint32_t rank, const int32_t *dimensions) {
  return udononnxBindingBind(session, output_index, buffer_slot,
                             buffer_generation, byte_offset, byte_length,
                             data_type, rank, dimensions, 1);
}

udononnxStatus_t
udononnxBindingStateCreate(void *state_storage, uint32_t state_storage_bytes,
                           uint32_t model_index, void *input_bindings,
                           uint32_t input_binding_bytes, void *output_bindings,
                           uint32_t output_binding_bytes) {
  struct udononnxBindingState *state;
  const udononnxModel_t *model = 0;
  cl_platform_id platform = 0;
  cl_device_id device = 0;
  cl_int error = CL_SUCCESS;
  udononnxStatus_t status;
  if (!state_storage ||
      state_storage_bytes < sizeof(struct udononnxBindingState) ||
      udononnxBindingStateGet(state_storage))
    return UDONONNX_STATUS_INVALID_VALUE;
  status = udononnxGetModel(model_index, &model);
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  if ((model->input_count &&
       (!input_bindings ||
        model->input_count > UINT32_MAX / sizeof(udononnxTensor_t) ||
        model->input_count * sizeof(udononnxTensor_t) > input_binding_bytes)) ||
      (model->output_count &&
       (!output_bindings ||
        model->output_count > UINT32_MAX / sizeof(udononnxTensor_t) ||
        model->output_count * sizeof(udononnxTensor_t) > output_binding_bytes)))
    return UDONONNX_STATUS_INVALID_VALUE;
  if (clGetPlatformIDs(1, &platform, 0) != CL_SUCCESS ||
      clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, 0) != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  state = (struct udononnxBindingState *)state_storage;
  state->magic = 0;
  state->context = clCreateContext(0, 1, &device, 0, 0, &error);
  if (!state->context || error != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  state->queue = clCreateCommandQueue(state->context, device, 0, &error);
  if (!state->queue || error != CL_SUCCESS) {
    clReleaseContext(state->context);
    state->context = 0;
    return UDONONNX_STATUS_OPENCL_ERROR;
  }
  status = udononnxCreateSession(&state->session, model, state->context,
                                 (udononnxTensor_t *)input_bindings,
                                 (udononnxTensor_t *)output_bindings);
  if (status != UDONONNX_STATUS_SUCCESS) {
    clReleaseCommandQueue(state->queue);
    clReleaseContext(state->context);
    state->queue = 0;
    state->context = 0;
    return status;
  }
  state->magic = UDONONNX_BINDING_STATE_MAGIC;
  return UDONONNX_STATUS_SUCCESS;
}

udononnxStatus_t udononnxBindingStateBindInput(
    void *state_storage, uint32_t input_index, uint32_t buffer_slot,
    uint32_t buffer_generation, uint32_t byte_offset, uint32_t byte_length,
    udonclDataType_t data_type, uint32_t rank, const int32_t *dimensions) {
  struct udononnxBindingState *state = udononnxBindingStateGet(state_storage);
  return state ? udononnxBindingBindInput(&state->session, input_index,
                                          buffer_slot, buffer_generation,
                                          byte_offset, byte_length, data_type,
                                          rank, dimensions)
               : UDONONNX_STATUS_NOT_INITIALIZED;
}

udononnxStatus_t udononnxBindingStateBindOutput(
    void *state_storage, uint32_t output_index, uint32_t buffer_slot,
    uint32_t buffer_generation, uint32_t byte_offset, uint32_t byte_length,
    udonclDataType_t data_type, uint32_t rank, const int32_t *dimensions) {
  struct udononnxBindingState *state = udononnxBindingStateGet(state_storage);
  return state ? udononnxBindingBindOutput(&state->session, output_index,
                                           buffer_slot, buffer_generation,
                                           byte_offset, byte_length, data_type,
                                           rank, dimensions)
               : UDONONNX_STATUS_NOT_INITIALIZED;
}

udononnxStatus_t
udononnxBindingStateAllocateBuffer(void *state_storage, uint32_t output,
                                   uint32_t byte_capacity,
                                   udonclBufferHandle_t *buffer) {
  struct udononnxBindingState *state = udononnxBindingStateGet(state_storage);
  udonclStatus_t status;
  if (!state)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  if (output > 1u || !byte_capacity || !buffer)
    return UDONONNX_STATUS_INVALID_VALUE;
  status = udonclDeviceBufferCreate(
      state->context, output ? CL_MEM_WRITE_ONLY : CL_MEM_READ_ONLY,
      byte_capacity, buffer);
  if (status == UDONCL_STATUS_OUT_OF_RESOURCES)
    return UDONONNX_STATUS_OPENCL_ERROR;
  return status == UDONCL_STATUS_SUCCESS ? UDONONNX_STATUS_SUCCESS
                                         : UDONONNX_STATUS_INVALID_VALUE;
}

udononnxStatus_t
udononnxBindingStateBindTensor(void *state_storage, uint32_t output,
                               uint32_t tensor_index, uint32_t buffer_slot,
                               uint32_t buffer_generation, uint32_t byte_offset,
                               uint32_t byte_length, udonclDataType_t data_type,
                               uint32_t rank, const int32_t *dimensions) {
  struct udononnxBindingState *state = udononnxBindingStateGet(state_storage);
  if (!state || output > 1u)
    return state ? UDONONNX_STATUS_INVALID_VALUE
                 : UDONONNX_STATUS_NOT_INITIALIZED;
  return output ? udononnxBindingBindOutput(&state->session, tensor_index,
                                            buffer_slot, buffer_generation,
                                            byte_offset, byte_length, data_type,
                                            rank, dimensions)
                : udononnxBindingBindInput(&state->session, tensor_index,
                                           buffer_slot, buffer_generation,
                                           byte_offset, byte_length, data_type,
                                           rank, dimensions);
}

udononnxStatus_t
udononnxBindingStateEnqueueInitialize(void *state_storage, uint32_t wait_count,
                                      const cl_event *wait_list,
                                      cl_event *completion) {
  struct udononnxBindingState *state = udononnxBindingStateGet(state_storage);
  return state ? udononnxEnqueueInitialize(&state->session, state->queue,
                                           wait_count, wait_list, completion)
               : UDONONNX_STATUS_NOT_INITIALIZED;
}

udononnxStatus_t udononnxBindingStateEnqueueRun(void *state_storage,
                                                uint32_t wait_count,
                                                const cl_event *wait_list,
                                                cl_event *completion) {
  struct udononnxBindingState *state = udononnxBindingStateGet(state_storage);
  return state ? udononnxEnqueueRun(&state->session, state->queue, wait_count,
                                    wait_list, completion)
               : UDONONNX_STATUS_NOT_INITIALIZED;
}

udononnxStatus_t udononnxBindingStateCompleteEvent(void *state_storage,
                                                   cl_event completion,
                                                   udononnxStatus_t result) {
  struct udononnxBindingState *state = udononnxBindingStateGet(state_storage);
  return state
             ? udononnxSessionCompleteEvent(&state->session, completion, result)
             : UDONONNX_STATUS_NOT_INITIALIZED;
}

udononnxStatus_t udononnxBindingStateGetContext(void *state_storage,
                                                cl_context *context) {
  struct udononnxBindingState *state = udononnxBindingStateGet(state_storage);
  if (!state)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  if (!context)
    return UDONONNX_STATUS_INVALID_VALUE;
  *context = state->context;
  return UDONONNX_STATUS_SUCCESS;
}

udononnxStatus_t udononnxBindingStateGetCommandQueue(void *state_storage,
                                                     cl_command_queue *queue) {
  struct udononnxBindingState *state = udononnxBindingStateGet(state_storage);
  if (!state)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  if (!queue)
    return UDONONNX_STATUS_INVALID_VALUE;
  *queue = state->queue;
  return UDONONNX_STATUS_SUCCESS;
}

udononnxStatus_t udononnxBindingStateDispose(void *state_storage) {
  struct udononnxBindingState *state = udononnxBindingStateGet(state_storage);
  udononnxStatus_t status;
  cl_int queue_status, context_status;
  if (!state)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  status = udononnxSessionDispose(&state->session);
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  queue_status = clReleaseCommandQueue(state->queue);
  context_status = clReleaseContext(state->context);
  state->queue = 0;
  state->context = 0;
  state->magic = 0;
  return queue_status == CL_SUCCESS && context_status == CL_SUCCESS
             ? UDONONNX_STATUS_SUCCESS
             : UDONONNX_STATUS_OPENCL_ERROR;
}
