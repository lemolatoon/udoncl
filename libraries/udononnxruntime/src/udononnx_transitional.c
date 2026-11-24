/* TRANSITIONAL, NOT PUBLIC API: removed in stage 3.
 *
 * The runtime side of the private sequence-state binding of a model library
 * built with TRANSITIONAL_SEQUENCE_STATE (udononnx_transitional.h). */
#include "udononnx_transitional.h"

#include <limits.h>

#if defined(__SPIR__)
_Static_assert(sizeof(udononnxTensorMetadata_t) ==
                   UDONCL_UDON_ABI_ONNX_TENSOR_METADATA_BYTES,
               "Udon ONNX tensor metadata ABI size drifted");
_Static_assert(offsetof(udononnxTensorMetadata_t, name) ==
                   UDONCL_UDON_ABI_ONNX_TENSOR_METADATA_NAME_OFFSET,
               "Udon ONNX tensor metadata name offset drifted");
_Static_assert(offsetof(udononnxTensorMetadata_t, data_type) ==
                   UDONCL_UDON_ABI_ONNX_TENSOR_METADATA_DATA_TYPE_OFFSET,
               "Udon ONNX tensor metadata dtype offset drifted");
_Static_assert(offsetof(udononnxTensorMetadata_t, rank) ==
                   UDONCL_UDON_ABI_ONNX_TENSOR_METADATA_RANK_OFFSET,
               "Udon ONNX tensor metadata rank offset drifted");
_Static_assert(offsetof(udononnxTensorMetadata_t, dimensions) ==
                   UDONCL_UDON_ABI_ONNX_TENSOR_METADATA_DIMENSIONS_OFFSET,
               "Udon ONNX tensor metadata dimensions offset drifted");
_Static_assert(offsetof(udononnxTensorMetadata_t, dimension_bounds) ==
                   UDONCL_UDON_ABI_ONNX_TENSOR_METADATA_DIMENSION_BOUNDS_OFFSET,
               "Udon ONNX tensor metadata bounds offset drifted");
_Static_assert(sizeof(udononnxModel_t) == UDONCL_UDON_ABI_ONNX_MODEL_BYTES,
               "Udon ONNX model ABI size drifted");
_Static_assert(offsetof(udononnxModel_t, name) ==
                   UDONCL_UDON_ABI_ONNX_MODEL_NAME_OFFSET,
               "Udon ONNX model name offset drifted");
_Static_assert(offsetof(udononnxModel_t, graph_id) ==
                   UDONCL_UDON_ABI_ONNX_MODEL_GRAPH_ID_OFFSET,
               "Udon ONNX model graph id offset drifted");
_Static_assert(offsetof(udononnxModel_t, inputs) ==
                   UDONCL_UDON_ABI_ONNX_MODEL_INPUTS_OFFSET,
               "Udon ONNX model inputs offset drifted");
_Static_assert(offsetof(udononnxModel_t, input_count) ==
                   UDONCL_UDON_ABI_ONNX_MODEL_INPUT_COUNT_OFFSET,
               "Udon ONNX model input count offset drifted");
_Static_assert(offsetof(udononnxModel_t, outputs) ==
                   UDONCL_UDON_ABI_ONNX_MODEL_OUTPUTS_OFFSET,
               "Udon ONNX model outputs offset drifted");
_Static_assert(offsetof(udononnxModel_t, output_count) ==
                   UDONCL_UDON_ABI_ONNX_MODEL_OUTPUT_COUNT_OFFSET,
               "Udon ONNX model output count offset drifted");
_Static_assert(sizeof(udononnxTensor_t) == UDONCL_UDON_ABI_ONNX_TENSOR_BYTES,
               "Udon ONNX tensor ABI size drifted");
_Static_assert(offsetof(udononnxTensor_t, storage) ==
                   UDONCL_UDON_ABI_ONNX_TENSOR_STORAGE_OFFSET,
               "Udon ONNX tensor storage offset drifted");
_Static_assert(offsetof(udononnxTensor_t, rank) ==
                   UDONCL_UDON_ABI_ONNX_TENSOR_RANK_OFFSET,
               "Udon ONNX tensor rank offset drifted");
_Static_assert(offsetof(udononnxTensor_t, dimensions) ==
                   UDONCL_UDON_ABI_ONNX_TENSOR_DIMENSIONS_OFFSET,
               "Udon ONNX tensor dimensions offset drifted");
_Static_assert(sizeof(((udononnxTensor_t *)0)->dimensions[0]) ==
                   UDONCL_UDON_ABI_ONNX_TENSOR_DIMENSION_STRIDE,
               "Udon ONNX tensor dimension stride drifted");
#endif

#define UDONONNX_BINDING_STATE_MAGIC 0x554f4e58u

static uint32_t udononnxNextGeneration(uint32_t generation) {
  ++generation;
  return generation ? generation : 1u;
}

static udononnxStatus_t udononnxElementCount(uint32_t rank,
                                             const int64_t *dimensions,
                                             size_t *elements) {
  uint32_t axis;
  size_t count = 1;
  if (!elements || rank > UDONONNX_MAX_TENSOR_RANK || (rank && !dimensions))
    return UDONONNX_STATUS_INVALID_VALUE;
  for (axis = 0; axis < rank; ++axis) {
    size_t dimension;
    uint64_t product;
    if (dimensions[axis] < 0 || (uint64_t)dimensions[axis] > SIZE_MAX)
      return UDONONNX_STATUS_INVALID_VALUE;
    dimension = (size_t)dimensions[axis];
    product = (uint64_t)count * (uint64_t)dimension;
    if (product > SIZE_MAX)
      return UDONONNX_STATUS_INVALID_VALUE;
    count = (size_t)product;
  }
  *elements = count;
  return UDONONNX_STATUS_SUCCESS;
}

static udononnxStatus_t
udononnxValidateTensorPointer(const udononnxTensor_t *tensor) {
  cl_mem memory;
  size_t byte_offset, byte_length, elements, element_bytes;
  udonclStatus_t storage_status;
  if (!tensor)
    return UDONONNX_STATUS_INVALID_VALUE;
  udononnxStatus_t status =
      udononnxElementCount(tensor->rank, tensor->dimensions, &elements);
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  element_bytes = udonclDataTypeSize(tensor->storage.data_type);
  if (!element_bytes || (uint64_t)elements * element_bytes > SIZE_MAX ||
      tensor->storage.byte_length != elements * element_bytes)
    return UDONONNX_STATUS_TYPE_MISMATCH;
  storage_status = udonclBufferViewResolve(tensor->storage, &memory,
                                           &byte_offset, &byte_length);
  if (storage_status == UDONCL_STATUS_STALE_HANDLE)
    return UDONONNX_STATUS_STALE_HANDLE;
  if (storage_status != UDONCL_STATUS_SUCCESS)
    return UDONONNX_STATUS_INVALID_VALUE;
  return UDONONNX_STATUS_SUCCESS;
}

static udononnxStatus_t
udononnxValidateMetadata(const udononnxTensorMetadata_t *metadata) {
  uint32_t axis;
  if (!metadata || !metadata->name || !metadata->name[0] ||
      (metadata->rank && !metadata->dimensions) ||
      metadata->rank > UDONONNX_MAX_TENSOR_RANK ||
      !udonclDataTypeSize(metadata->data_type))
    return UDONONNX_STATUS_INVALID_VALUE;
  for (axis = 0; axis < metadata->rank; ++axis) {
    if (metadata->dimensions[axis] >= 0)
      continue;
    if (metadata->dimensions[axis] != -1 || !metadata->dimension_bounds ||
        metadata->dimension_bounds[axis] <= 0)
      return UDONONNX_STATUS_INVALID_VALUE;
  }
  return UDONONNX_STATUS_SUCCESS;
}

static udononnxStatus_t
udononnxValidateBinding(const udononnxTensorMetadata_t *metadata,
                        const udononnxTensor_t *tensor, cl_context context) {
  uint32_t axis;
  cl_context buffer_context = 0;
  udononnxStatus_t status = udononnxValidateMetadata(metadata);
  udonclStatus_t buffer_status;
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  status = udononnxValidateTensorPointer(tensor);
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  if (tensor->storage.data_type != metadata->data_type ||
      tensor->rank != metadata->rank)
    return UDONONNX_STATUS_TYPE_MISMATCH;
  for (axis = 0; axis < tensor->rank; ++axis) {
    if (metadata->dimensions[axis] > 0 &&
        tensor->dimensions[axis] != metadata->dimensions[axis])
      return UDONONNX_STATUS_TYPE_MISMATCH;
    if (metadata->dimensions[axis] == -1 &&
        tensor->dimensions[axis] > metadata->dimension_bounds[axis])
      return UDONONNX_STATUS_INVALID_VALUE;
  }
  buffer_status =
      udonclDeviceBufferGetContext(tensor->storage.buffer, &buffer_context);
  if (buffer_status == UDONCL_STATUS_STALE_HANDLE)
    return UDONONNX_STATUS_STALE_HANDLE;
  if (buffer_status != UDONCL_STATUS_SUCCESS)
    return UDONONNX_STATUS_INVALID_VALUE;
  return buffer_context == context ? UDONONNX_STATUS_SUCCESS
                                   : UDONONNX_STATUS_CONTEXT_MISMATCH;
}

/* Binding versions are unique across sessions: the last one handed out. */
static uint32_t udononnx_last_binding_version;

static uint32_t udononnxNextBindingVersion(void) {
  udononnx_last_binding_version =
      udononnxNextGeneration(udononnx_last_binding_version);
  return udononnx_last_binding_version;
}

static void udononnxClearBinding(udononnxTensor_t *tensor) {
  uint32_t axis;
  tensor->storage.buffer.slot = 0;
  tensor->storage.buffer.generation = 0;
  tensor->storage.byte_offset = 0;
  tensor->storage.byte_length = 0;
  tensor->storage.data_type = 0;
  tensor->rank = 0;
  for (axis = 0; axis < UDONONNX_MAX_TENSOR_RANK; ++axis)
    tensor->dimensions[axis] = 0;
}

static udononnxStatus_t
udononnxSessionCreate(udononnxTransitionalSession_t *session,
                      const udononnxModel_t *model, cl_context context,
                      udononnxTensor_t *input_bindings,
                      udononnxTensor_t *output_bindings) {
  size_t index;
  uint32_t generation;
  if (!session || !model || !model->name || !model->name[0] || !context ||
      model->input_count > INT_MAX || model->output_count > INT_MAX ||
      (model->input_count && (!model->inputs || !input_bindings)) ||
      (model->output_count && (!model->outputs || !output_bindings)))
    return UDONONNX_STATUS_INVALID_VALUE;
  for (index = 0; index < model->input_count; ++index)
    if (udononnxValidateMetadata(&model->inputs[index]) !=
        UDONONNX_STATUS_SUCCESS)
      return UDONONNX_STATUS_INVALID_VALUE;
  for (index = 0; index < model->output_count; ++index)
    if (udononnxValidateMetadata(&model->outputs[index]) !=
        UDONONNX_STATUS_SUCCESS)
      return UDONONNX_STATUS_INVALID_VALUE;
  generation = udononnxNextGeneration(session->generation);
  session->model = model;
  session->context = context;
  session->input_bindings = input_bindings;
  session->output_bindings = output_bindings;
  session->generation = generation;
  session->operation_generation = 0;
  session->last_status = UDONONNX_STATUS_SUCCESS;
  session->active_event = 0;
  session->state = UDONONNX_SESSION_CREATED;
  session->bindings_validated = 0;
  session->validated_buffer_epoch = 0;
  session->binding_version = udononnxNextBindingVersion();
  for (index = 0; index < model->input_count; ++index)
    udononnxClearBinding(&input_bindings[index]);
  for (index = 0; index < model->output_count; ++index)
    udononnxClearBinding(&output_bindings[index]);
  return UDONONNX_STATUS_SUCCESS;
}

static void
udononnxSessionInvalidateBindings(udononnxTransitionalSession_t *session) {
  session->bindings_validated = 0;
  session->binding_version = udononnxNextBindingVersion();
}

uint32_t
udononnxSessionBindingVersion(const udononnxTransitionalSession_t *session) {
  return session ? session->binding_version : 0;
}

static udononnxStatus_t
udononnxValidateQueue(udononnxTransitionalSession_t *session,
                      cl_command_queue queue) {
  cl_context queue_context = 0;
  if (!queue)
    return UDONONNX_STATUS_INVALID_VALUE;
  if (clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(queue_context),
                            &queue_context, 0) != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  return queue_context == session->context ? UDONONNX_STATUS_SUCCESS
                                           : UDONONNX_STATUS_CONTEXT_MISMATCH;
}

static int udononnxTensorIsBound(const udononnxTensor_t *tensor) {
  return tensor && tensor->storage.buffer.slot &&
         tensor->storage.buffer.generation;
}

static udononnxStatus_t
udononnxSessionResolve(const udononnxTransitionalSession_t *session,
                       size_t index, const udononnxTensor_t *bindings,
                       size_t count, cl_mem *memory, size_t *byte_offset,
                       size_t *byte_length) {
  udonclStatus_t status;
  if (!session || !memory || !byte_offset || !byte_length)
    return UDONONNX_STATUS_INVALID_VALUE;
  if (session->state == UDONONNX_SESSION_DISPOSED)
    return UDONONNX_STATUS_DISPOSED;
  if (session->state != UDONONNX_SESSION_READY &&
      session->state != UDONONNX_SESSION_RUNNING)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  if (index >= count)
    return UDONONNX_STATUS_INVALID_VALUE;
  if (!udononnxTensorIsBound(&bindings[index]))
    return UDONONNX_STATUS_MISSING_BINDING;
  status = udonclBufferViewResolve(bindings[index].storage, memory, byte_offset,
                                   byte_length);
  if (status == UDONCL_STATUS_STALE_HANDLE)
    return UDONONNX_STATUS_STALE_HANDLE;
  return status == UDONCL_STATUS_SUCCESS ? UDONONNX_STATUS_SUCCESS
                                         : UDONONNX_STATUS_INVALID_VALUE;
}

udononnxStatus_t
udononnxSessionResolveInput(const udononnxTransitionalSession_t *session,
                            size_t index, cl_mem *memory, size_t *byte_offset,
                            size_t *byte_length) {
  if (!session || !session->model)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  return udononnxSessionResolve(session, index, session->input_bindings,
                                session->model->input_count, memory,
                                byte_offset, byte_length);
}

udononnxStatus_t
udononnxSessionResolveOutput(const udononnxTransitionalSession_t *session,
                             size_t index, cl_mem *memory, size_t *byte_offset,
                             size_t *byte_length) {
  if (!session || !session->model)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  return udononnxSessionResolve(session, index, session->output_bindings,
                                session->model->output_count, memory,
                                byte_offset, byte_length);
}

static int udononnxRangesOverlap(const udonclBufferView_t *left,
                                 const udonclBufferView_t *right) {
  size_t left_end, right_end;
  if (left->buffer.slot != right->buffer.slot ||
      left->buffer.generation != right->buffer.generation)
    return 0;
  left_end = left->byte_offset + left->byte_length;
  right_end = right->byte_offset + right->byte_length;
  return left->byte_offset < right_end && right->byte_offset < left_end;
}

/* The part of a binding that can change without a bind: the bound buffer
 * may have been destroyed. A live handle keeps its size and context, so
 * shape, type, range and overlap need no second look. */
static udononnxStatus_t
udononnxValidateBindingLiveness(const udononnxTensor_t *tensor,
                                cl_context context) {
  cl_context buffer_context = 0;
  udonclStatus_t status =
      udonclDeviceBufferGetContext(tensor->storage.buffer, &buffer_context);
  if (status == UDONCL_STATUS_STALE_HANDLE)
    return UDONONNX_STATUS_STALE_HANDLE;
  if (status != UDONCL_STATUS_SUCCESS)
    return UDONONNX_STATUS_INVALID_VALUE;
  return buffer_context == context ? UDONONNX_STATUS_SUCCESS
                                   : UDONONNX_STATUS_CONTEXT_MISMATCH;
}

static udononnxStatus_t udononnxValidateRunBindingLiveness(
    const udononnxTransitionalSession_t *session) {
  size_t index;
  udononnxStatus_t status;
  for (index = 0; index < session->model->input_count; ++index) {
    status = udononnxValidateBindingLiveness(&session->input_bindings[index],
                                             session->context);
    if (status != UDONONNX_STATUS_SUCCESS)
      return status;
  }
  for (index = 0; index < session->model->output_count; ++index) {
    if (!udononnxTensorIsBound(&session->output_bindings[index]))
      continue;
    status = udononnxValidateBindingLiveness(&session->output_bindings[index],
                                             session->context);
    if (status != UDONONNX_STATUS_SUCCESS)
      return status;
  }
  return UDONONNX_STATUS_SUCCESS;
}

static udononnxStatus_t
udononnxValidateRunBindings(udononnxTransitionalSession_t *session) {
  size_t input, output, bound_outputs = 0;
  udononnxStatus_t status;
  uint32_t epoch = udonclDeviceBufferEpoch();
  if (session->bindings_validated) {
    /* No handle became stale since the table was last found alive. */
    if (session->validated_buffer_epoch == epoch)
      return UDONONNX_STATUS_SUCCESS;
    status = udononnxValidateRunBindingLiveness(session);
    if (status == UDONONNX_STATUS_SUCCESS)
      session->validated_buffer_epoch = epoch;
    return status;
  }
  for (input = 0; input < session->model->input_count; ++input) {
    if (!udononnxTensorIsBound(&session->input_bindings[input]))
      return UDONONNX_STATUS_MISSING_BINDING;
    status = udononnxValidateBinding(&session->model->inputs[input],
                                     &session->input_bindings[input],
                                     session->context);
    if (status != UDONONNX_STATUS_SUCCESS)
      return status;
  }
  for (output = 0; output < session->model->output_count; ++output) {
    if (!udononnxTensorIsBound(&session->output_bindings[output]))
      continue;
    ++bound_outputs;
    status = udononnxValidateBinding(&session->model->outputs[output],
                                     &session->output_bindings[output],
                                     session->context);
    if (status != UDONONNX_STATUS_SUCCESS)
      return status;
    for (input = 0; input < session->model->input_count; ++input)
      if (udononnxRangesOverlap(&session->input_bindings[input].storage,
                                &session->output_bindings[output].storage))
        return UDONONNX_STATUS_INVALID_VALUE;
    for (input = 0; input < output; ++input)
      if (udononnxTensorIsBound(&session->output_bindings[input]) &&
          udononnxRangesOverlap(&session->output_bindings[input].storage,
                                &session->output_bindings[output].storage))
        return UDONONNX_STATUS_INVALID_VALUE;
  }
  if (session->model->output_count && !bound_outputs)
    return UDONONNX_STATUS_MISSING_BINDING;
  session->bindings_validated = 1;
  session->validated_buffer_epoch = epoch;
  return UDONONNX_STATUS_SUCCESS;
}

static udononnxStatus_t
udononnxRecordFailure(udononnxTransitionalSession_t *session,
                      udononnxStatus_t status) {
  if (session)
    session->last_status = status;
  return status;
}

static udononnxStatus_t
udononnxSessionEnqueueRun(udononnxTransitionalSession_t *session,
                          cl_command_queue queue, cl_uint wait_count,
                          const cl_event *wait_list, cl_event *completion) {
  udononnxStatus_t status;
  if (!session || !completion || (!!wait_count != !!wait_list))
    return udononnxRecordFailure(session, UDONONNX_STATUS_INVALID_VALUE);
  *completion = 0;
  if (session->state == UDONONNX_SESSION_DISPOSED)
    return udononnxRecordFailure(session, UDONONNX_STATUS_DISPOSED);
  if (session->state == UDONONNX_SESSION_RUNNING)
    return udononnxRecordFailure(session, UDONONNX_STATUS_BUSY);
  if (session->state != UDONONNX_SESSION_READY)
    return udononnxRecordFailure(session, UDONONNX_STATUS_NOT_INITIALIZED);
  status = udononnxValidateQueue(session, queue);
  if (status != UDONONNX_STATUS_SUCCESS)
    return udononnxRecordFailure(session, status);
  status = udononnxValidateRunBindings(session);
  if (status != UDONONNX_STATUS_SUCCESS)
    return udononnxRecordFailure(session, status);
  status = udononnxGeneratedEnqueueRun(session->model->graph_id, session, queue,
                                       wait_count, wait_list, completion);
  if (status != UDONONNX_STATUS_SUCCESS || !*completion)
    return udononnxRecordFailure(session,
                                 status == UDONONNX_STATUS_SUCCESS
                                     ? UDONONNX_STATUS_EXECUTION_FAILURE
                                     : status);
  session->operation_generation =
      udononnxNextGeneration(session->operation_generation);
  session->active_event = *completion;
  session->state = UDONONNX_SESSION_RUNNING;
  session->last_status = UDONONNX_STATUS_BUSY;
  return UDONONNX_STATUS_SUCCESS;
}

static udononnxStatus_t
udononnxSessionCompleteEvent(udononnxTransitionalSession_t *session,
                             cl_event completion, udononnxStatus_t result) {
  if (!session || !completion)
    return UDONONNX_STATUS_INVALID_VALUE;
  if (session->state != UDONONNX_SESSION_RUNNING ||
      completion != session->active_event)
    return UDONONNX_STATUS_STALE_HANDLE;
  session->active_event = 0;
  session->last_status = result;
  session->state = UDONONNX_SESSION_READY;
  return result;
}

static udononnxStatus_t
udononnxSessionDispose(udononnxTransitionalSession_t *session) {
  udononnxStatus_t status;
  if (!session)
    return UDONONNX_STATUS_INVALID_VALUE;
  if (session->state == UDONONNX_SESSION_DISPOSED)
    return UDONONNX_STATUS_SUCCESS;
  if (session->state == UDONONNX_SESSION_RUNNING)
    return UDONONNX_STATUS_BUSY;
  if (session->model) {
    status = udononnxGeneratedDispose(session->model->graph_id, session);
    if (status != UDONONNX_STATUS_SUCCESS)
      return udononnxRecordFailure(session, status);
  }
  session->generation = udononnxNextGeneration(session->generation);
  session->active_event = 0;
  session->state = UDONONNX_SESSION_DISPOSED;
  session->last_status = UDONONNX_STATUS_SUCCESS;
  return UDONONNX_STATUS_SUCCESS;
}

struct udononnxBindingState {
  uint32_t magic;
  cl_context context;
  cl_command_queue queue;
  udononnxTransitionalSession_t session;
};

static struct udononnxBindingState *
udononnxBindingStateGet(void *state_storage) {
  struct udononnxBindingState *state =
      (struct udononnxBindingState *)state_storage;
  return state && state->magic == UDONONNX_BINDING_STATE_MAGIC ? state : 0;
}

udononnxStatus_t
udononnxBindingStateCreate(void *state_storage, uint32_t state_storage_bytes,
                           cl_context context, cl_command_queue queue,
                           uint32_t model_index, void *input_bindings,
                           uint32_t input_binding_bytes, void *output_bindings,
                           uint32_t output_binding_bytes) {
  struct udononnxBindingState *state;
  const udononnxModel_t *model = 0;
  cl_context queue_context = 0;
  udononnxStatus_t status;
  if (!state_storage || !context || !queue ||
      state_storage_bytes < sizeof(struct udononnxBindingState) ||
      udononnxBindingStateGet(state_storage))
    return UDONONNX_STATUS_INVALID_VALUE;
  if (clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(queue_context),
                            &queue_context, 0) != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  if (queue_context != context)
    return UDONONNX_STATUS_CONTEXT_MISMATCH;
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
  state = (struct udononnxBindingState *)state_storage;
  state->magic = 0;
  state->context = context;
  state->queue = queue;
  status = udononnxSessionCreate(&state->session, model, state->context,
                                 (udononnxTensor_t *)input_bindings,
                                 (udononnxTensor_t *)output_bindings);
  if (status != UDONONNX_STATUS_SUCCESS) {
    state->queue = 0;
    state->context = 0;
    return status;
  }
  state->magic = UDONONNX_BINDING_STATE_MAGIC;
  return UDONONNX_STATUS_SUCCESS;
}

udononnxStatus_t udononnxBindingStateCreateSession(void *state_storage) {
  struct udononnxBindingState *state = udononnxBindingStateGet(state_storage);
  udononnxStatus_t status;
  if (!state)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  if (state->session.state != UDONONNX_SESSION_CREATED)
    return UDONONNX_STATUS_INVALID_STATE;
  /* Under the static Unity lowering the session's buffers are assigned
   * Textures, so creating them enqueues nothing. */
  status = udononnxGeneratedCreate(state->session.model->graph_id,
                                   &state->session, state->queue);
  if (status != UDONONNX_STATUS_SUCCESS) {
    udononnxGeneratedDispose(state->session.model->graph_id, &state->session);
    state->session.last_status = status;
    return status;
  }
  state->session.state = UDONONNX_SESSION_READY;
  return UDONONNX_STATUS_SUCCESS;
}

udononnxStatus_t
udononnxBindingStateAllocateBuffer(void *state_storage, uint32_t access,
                                   uint32_t byte_capacity,
                                   udonclBufferHandle_t *buffer) {
  struct udononnxBindingState *state = udononnxBindingStateGet(state_storage);
  udonclStatus_t status;
  if (!state)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  if (access > 2u || !byte_capacity || !buffer)
    return UDONONNX_STATUS_INVALID_VALUE;
  status = udonclDeviceBufferCreate(state->context,
                                    access == 2u ? CL_MEM_READ_WRITE
                                    : access     ? CL_MEM_WRITE_ONLY
                                                 : CL_MEM_READ_ONLY,
                                    byte_capacity, buffer);
  if (status == UDONCL_STATUS_OUT_OF_RESOURCES)
    return UDONONNX_STATUS_OPENCL_ERROR;
  return status == UDONCL_STATUS_SUCCESS ? UDONONNX_STATUS_SUCCESS
                                         : UDONONNX_STATUS_INVALID_VALUE;
}

udononnxStatus_t udononnxBindingStateEnqueueRun(void *state_storage,
                                                uint32_t wait_count,
                                                const cl_event *wait_list,
                                                cl_event *completion) {
  struct udononnxBindingState *state = udononnxBindingStateGet(state_storage);
  return state ? udononnxSessionEnqueueRun(&state->session, state->queue,
                                           wait_count, wait_list, completion)
               : UDONONNX_STATUS_NOT_INITIALIZED;
}

udononnxStatus_t
udononnxBindingStateGetSession(void *state_storage,
                               udononnxTransitionalSession_t **session) {
  struct udononnxBindingState *state = udononnxBindingStateGet(state_storage);
  if (!state)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  if (!session)
    return UDONONNX_STATUS_INVALID_VALUE;
  *session = &state->session;
  return UDONONNX_STATUS_SUCCESS;
}

udononnxStatus_t udononnxBindingStateBeginStaticBind(void *state_storage) {
  struct udononnxBindingState *state = udononnxBindingStateGet(state_storage);
  udononnxStatus_t status;
  if (!state)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  if (state->session.state == UDONONNX_SESSION_DISPOSED)
    status = UDONONNX_STATUS_DISPOSED;
  else if (state->session.state == UDONONNX_SESSION_RUNNING)
    status = UDONONNX_STATUS_BUSY;
  else if (state->session.state != UDONONNX_SESSION_READY)
    status = UDONONNX_STATUS_NOT_INITIALIZED;
  else
    status = UDONONNX_STATUS_SUCCESS;
  /* The generated bind-all writes the binding tables itself. */
  udononnxSessionInvalidateBindings(&state->session);
  state->session.last_status = status;
  return status;
}

udononnxStatus_t
udononnxBindingStateCompleteStaticBind(void *state_storage,
                                       udononnxStatus_t status) {
  struct udononnxBindingState *state = udononnxBindingStateGet(state_storage);
  if (!state)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  udononnxSessionInvalidateBindings(&state->session);
  state->session.last_status = status;
  return status;
}

udononnxStatus_t udononnxBindingStateResolveInput(void *state_storage,
                                                  uint32_t input_index,
                                                  cl_mem *memory,
                                                  size_t *byte_offset,
                                                  size_t *byte_length) {
  struct udononnxBindingState *state = udononnxBindingStateGet(state_storage);
  return state ? udononnxSessionResolveInput(&state->session, input_index,
                                             memory, byte_offset, byte_length)
               : UDONONNX_STATUS_NOT_INITIALIZED;
}

udononnxStatus_t udononnxBindingStateResolveOutput(void *state_storage,
                                                   uint32_t output_index,
                                                   cl_mem *memory,
                                                   size_t *byte_offset,
                                                   size_t *byte_length) {
  struct udononnxBindingState *state = udononnxBindingStateGet(state_storage);
  return state ? udononnxSessionResolveOutput(&state->session, output_index,
                                              memory, byte_offset, byte_length)
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
  if (!state)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  status = udononnxSessionDispose(&state->session);
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  state->queue = 0;
  state->context = 0;
  state->magic = 0;
  return UDONONNX_STATUS_SUCCESS;
}
