#include "udononnxruntime.h"

#include <limits.h>

static uint32_t udononnxNextGeneration(uint32_t generation) {
  ++generation;
  return generation ? generation : 1u;
}

static int udononnxStringEqual(const char *left, const char *right) {
  if (!left || !right)
    return 0;
  while (*left && *left == *right) {
    ++left;
    ++right;
  }
  return *left == *right;
}

static udononnxStatus_t udononnxRecordFailure(udononnxSession_t *session,
                                              udononnxStatus_t status);

static void udononnxCopyBufferView(udonclBufferView_t *destination,
                                   const udonclBufferView_t *source) {
  destination->buffer.slot = source->buffer.slot;
  destination->buffer.generation = source->buffer.generation;
  destination->byte_offset = source->byte_offset;
  destination->byte_length = source->byte_length;
  destination->data_type = source->data_type;
}

static void udononnxCopyTensor(udononnxTensor_t *destination,
                               const udononnxTensor_t *source) {
  udononnxCopyBufferView(&destination->storage, &source->storage);
  destination->rank = source->rank;
  destination->dimensions[0] = source->dimensions[0];
  destination->dimensions[1] = source->dimensions[1];
  destination->dimensions[2] = source->dimensions[2];
  destination->dimensions[3] = source->dimensions[3];
  destination->dimensions[4] = source->dimensions[4];
  destination->dimensions[5] = source->dimensions[5];
  destination->dimensions[6] = source->dimensions[6];
  destination->dimensions[7] = source->dimensions[7];
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
    if (dimensions[axis] <= 0 || (uint64_t)dimensions[axis] > SIZE_MAX)
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

udononnxStatus_t udononnxCreateTensor(udonclBufferView_t storage, uint32_t rank,
                                      const int64_t *dimensions,
                                      udononnxTensor_t *tensor) {
  uint32_t axis;
  size_t elements, element_bytes;
  udononnxStatus_t status;
  if (!tensor)
    return UDONONNX_STATUS_INVALID_VALUE;
  status = udononnxElementCount(rank, dimensions, &elements);
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  element_bytes = udonclDataTypeSize(storage.data_type);
  if (!element_bytes || (uint64_t)elements * element_bytes > SIZE_MAX ||
      storage.byte_length != elements * element_bytes)
    return UDONONNX_STATUS_TYPE_MISMATCH;
  udononnxCopyBufferView(&tensor->storage, &storage);
  tensor->rank = rank;
  for (axis = 0; axis < UDONONNX_MAX_TENSOR_RANK; ++axis)
    tensor->dimensions[axis] = axis < rank ? dimensions[axis] : 0;
  return udononnxValidateTensor(*tensor);
}

udononnxStatus_t udononnxGetTensorElementCount(udononnxTensor_t tensor,
                                               size_t *elements) {
  return udononnxElementCount(tensor.rank, tensor.dimensions, elements);
}

udononnxStatus_t udononnxGetTensorDimension(udononnxTensor_t tensor,
                                            uint32_t axis, int64_t *dimension) {
  if (!dimension || axis >= tensor.rank ||
      tensor.rank > UDONONNX_MAX_TENSOR_RANK)
    return UDONONNX_STATUS_INVALID_VALUE;
  *dimension = tensor.dimensions[axis];
  return UDONONNX_STATUS_SUCCESS;
}

udononnxStatus_t udononnxGetTensorStorage(udononnxTensor_t tensor,
                                          udonclBufferView_t *storage) {
  udononnxStatus_t status;
  if (!storage)
    return UDONONNX_STATUS_INVALID_VALUE;
  status = udononnxValidateTensor(tensor);
  if (status == UDONONNX_STATUS_SUCCESS)
    udononnxCopyBufferView(storage, &tensor.storage);
  return status;
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

udononnxStatus_t udononnxValidateTensor(udononnxTensor_t tensor) {
  return udononnxValidateTensorPointer(&tensor);
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
    if (metadata->dimensions[axis] > 0)
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

udononnxStatus_t udononnxFindModel(const udononnxModel_t *models,
                                   size_t model_count, const char *name,
                                   const udononnxModel_t **model) {
  size_t index;
  if (!models || !model_count || !name || !model)
    return UDONONNX_STATUS_INVALID_VALUE;
  for (index = 0; index < model_count; ++index) {
    if (udononnxStringEqual(models[index].name, name)) {
      *model = &models[index];
      return UDONONNX_STATUS_SUCCESS;
    }
  }
  *model = 0;
  return UDONONNX_STATUS_INVALID_VALUE;
}

size_t udononnxGetModelCount(void) { return udononnxGeneratedGetModelCount(); }

udononnxStatus_t udononnxGetModel(size_t index, const udononnxModel_t **model) {
  if (!model)
    return UDONONNX_STATUS_INVALID_VALUE;
  *model = 0;
  if (index >= udononnxGeneratedGetModelCount())
    return UDONONNX_STATUS_INVALID_VALUE;
  return udononnxGeneratedGetModel(index, model);
}

udononnxStatus_t udononnxFindGeneratedModel(const char *name,
                                            const udononnxModel_t **model) {
  size_t index, count;
  if (!name || !model)
    return UDONONNX_STATUS_INVALID_VALUE;
  *model = 0;
  count = udononnxGeneratedGetModelCount();
  for (index = 0; index < count; ++index) {
    const udononnxModel_t *candidate = 0;
    udononnxStatus_t status = udononnxGeneratedGetModel(index, &candidate);
    if (status != UDONONNX_STATUS_SUCCESS)
      return status;
    if (candidate && udononnxStringEqual(candidate->name, name)) {
      *model = candidate;
      return UDONONNX_STATUS_SUCCESS;
    }
  }
  return UDONONNX_STATUS_INVALID_VALUE;
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

udononnxStatus_t udononnxCreateSession(udononnxSession_t *session,
                                       const udononnxModel_t *model,
                                       cl_context context,
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
  session->last_error = udononnxStatusString(UDONONNX_STATUS_SUCCESS);
  session->active_event = 0;
  session->state = UDONONNX_SESSION_CREATED;
  for (index = 0; index < model->input_count; ++index)
    udononnxClearBinding(&input_bindings[index]);
  for (index = 0; index < model->output_count; ++index)
    udononnxClearBinding(&output_bindings[index]);
  return UDONONNX_STATUS_SUCCESS;
}

static int udononnxSessionUsable(const udononnxSession_t *session) {
  return session && session->state != UDONONNX_SESSION_EMPTY &&
         session->state != UDONONNX_SESSION_DISPOSED;
}

size_t udononnxSessionGetInputCount(const udononnxSession_t *session) {
  return udononnxSessionUsable(session) ? session->model->input_count : 0;
}

size_t udononnxSessionGetOutputCount(const udononnxSession_t *session) {
  return udononnxSessionUsable(session) ? session->model->output_count : 0;
}

const udononnxTensorMetadata_t *
udononnxSessionGetInputMetadata(const udononnxSession_t *session,
                                size_t index) {
  return udononnxSessionUsable(session) && index < session->model->input_count
             ? &session->model->inputs[index]
             : 0;
}

const udononnxTensorMetadata_t *
udononnxSessionGetOutputMetadata(const udononnxSession_t *session,
                                 size_t index) {
  return udononnxSessionUsable(session) && index < session->model->output_count
             ? &session->model->outputs[index]
             : 0;
}

static int udononnxSessionFind(const udononnxTensorMetadata_t *metadata,
                               size_t count, const char *name) {
  size_t index;
  if (!metadata || !name)
    return -1;
  for (index = 0; index < count; ++index)
    if (udononnxStringEqual(metadata[index].name, name))
      return (int)index;
  return -1;
}

int udononnxSessionFindInput(const udononnxSession_t *session,
                             const char *name) {
  return udononnxSessionUsable(session)
             ? udononnxSessionFind(session->model->inputs,
                                   session->model->input_count, name)
             : -1;
}

int udononnxSessionFindOutput(const udononnxSession_t *session,
                              const char *name) {
  return udononnxSessionUsable(session)
             ? udononnxSessionFind(session->model->outputs,
                                   session->model->output_count, name)
             : -1;
}

static udononnxStatus_t
udononnxSessionBind(udononnxSession_t *session, size_t index,
                    udononnxTensor_t tensor,
                    const udononnxTensorMetadata_t *metadata, size_t count,
                    udononnxTensor_t *bindings) {
  udononnxStatus_t status;
  if (!session)
    return UDONONNX_STATUS_INVALID_VALUE;
  if (session->state == UDONONNX_SESSION_DISPOSED)
    return UDONONNX_STATUS_DISPOSED;
  if (session->state == UDONONNX_SESSION_INITIALIZING ||
      session->state == UDONONNX_SESSION_RUNNING)
    return UDONONNX_STATUS_BUSY;
  if (session->state != UDONONNX_SESSION_READY)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  if (index >= count)
    return UDONONNX_STATUS_INVALID_VALUE;
  status = udononnxValidateBinding(&metadata[index], &tensor, session->context);
  if (status == UDONONNX_STATUS_SUCCESS)
    udononnxCopyTensor(&bindings[index], &tensor);
  return status;
}

udononnxStatus_t udononnxSessionBindInput(udononnxSession_t *session,
                                          size_t index,
                                          udononnxTensor_t tensor) {
  udononnxStatus_t status;
  if (!session)
    return UDONONNX_STATUS_INVALID_VALUE;
  if (!session->model)
    return udononnxRecordFailure(session, UDONONNX_STATUS_NOT_INITIALIZED);
  status =
      udononnxSessionBind(session, index, tensor, session->model->inputs,
                          session->model->input_count, session->input_bindings);
  return udononnxRecordFailure(session, status);
}

udononnxStatus_t udononnxSessionBindOutput(udononnxSession_t *session,
                                           size_t index,
                                           udononnxTensor_t tensor) {
  udononnxStatus_t status;
  if (!session)
    return UDONONNX_STATUS_INVALID_VALUE;
  if (!session->model)
    return udononnxRecordFailure(session, UDONONNX_STATUS_NOT_INITIALIZED);
  status = udononnxSessionBind(session, index, tensor, session->model->outputs,
                               session->model->output_count,
                               session->output_bindings);
  return udononnxRecordFailure(session, status);
}

static udononnxStatus_t udononnxValidateQueue(udononnxSession_t *session,
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

static udononnxStatus_t udononnxSessionResolve(const udononnxSession_t *session,
                                               size_t index,
                                               const udononnxTensor_t *bindings,
                                               size_t count, cl_mem *memory,
                                               size_t *byte_offset,
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

udononnxStatus_t udononnxSessionResolveInput(const udononnxSession_t *session,
                                             size_t index, cl_mem *memory,
                                             size_t *byte_offset,
                                             size_t *byte_length) {
  if (!session || !session->model)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  return udononnxSessionResolve(session, index, session->input_bindings,
                                session->model->input_count, memory,
                                byte_offset, byte_length);
}

udononnxStatus_t udononnxSessionResolveOutput(const udononnxSession_t *session,
                                              size_t index, cl_mem *memory,
                                              size_t *byte_offset,
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

static udononnxStatus_t
udononnxValidateRunBindings(udononnxSession_t *session) {
  size_t input, output, bound_outputs = 0;
  udononnxStatus_t status;
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
  return UDONONNX_STATUS_SUCCESS;
}

static udononnxStatus_t udononnxRecordFailure(udononnxSession_t *session,
                                              udononnxStatus_t status) {
  if (session) {
    session->last_status = status;
    session->last_error = udononnxStatusString(status);
  }
  return status;
}

udononnxStatus_t udononnxEnqueueInitialize(udononnxSession_t *session,
                                           cl_command_queue queue,
                                           cl_uint wait_count,
                                           const cl_event *wait_list,
                                           cl_event *completion) {
  udononnxStatus_t status;
  if (!session || !completion || (!!wait_count != !!wait_list))
    return udononnxRecordFailure(session, UDONONNX_STATUS_INVALID_VALUE);
  *completion = 0;
  if (session->state == UDONONNX_SESSION_DISPOSED)
    return udononnxRecordFailure(session, UDONONNX_STATUS_DISPOSED);
  if (session->state == UDONONNX_SESSION_INITIALIZING ||
      session->state == UDONONNX_SESSION_RUNNING)
    return udononnxRecordFailure(session, UDONONNX_STATUS_BUSY);
  if (session->state != UDONONNX_SESSION_CREATED)
    return udononnxRecordFailure(session, UDONONNX_STATUS_INVALID_VALUE);
  status = udononnxValidateQueue(session, queue);
  if (status != UDONONNX_STATUS_SUCCESS)
    return udononnxRecordFailure(session, status);
  status = udononnxGeneratedEnqueueInitialize(session->model->graph_id, session,
                                              queue, wait_count, wait_list,
                                              completion);
  if (status != UDONONNX_STATUS_SUCCESS || !*completion)
    return udononnxRecordFailure(session,
                                 status == UDONONNX_STATUS_SUCCESS
                                     ? UDONONNX_STATUS_EXECUTION_FAILURE
                                     : status);
  session->operation_generation =
      udononnxNextGeneration(session->operation_generation);
  session->active_event = *completion;
  session->state = UDONONNX_SESSION_INITIALIZING;
  session->last_status = UDONONNX_STATUS_BUSY;
  session->last_error = udononnxStatusString(UDONONNX_STATUS_BUSY);
  return UDONONNX_STATUS_SUCCESS;
}

udononnxStatus_t udononnxEnqueueRun(udononnxSession_t *session,
                                    cl_command_queue queue, cl_uint wait_count,
                                    const cl_event *wait_list,
                                    cl_event *completion) {
  udononnxStatus_t status;
  if (!session || !completion || (!!wait_count != !!wait_list))
    return udononnxRecordFailure(session, UDONONNX_STATUS_INVALID_VALUE);
  *completion = 0;
  if (session->state == UDONONNX_SESSION_DISPOSED)
    return udononnxRecordFailure(session, UDONONNX_STATUS_DISPOSED);
  if (session->state == UDONONNX_SESSION_INITIALIZING ||
      session->state == UDONONNX_SESSION_RUNNING)
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
  session->last_error = udononnxStatusString(UDONONNX_STATUS_BUSY);
  return UDONONNX_STATUS_SUCCESS;
}

udononnxStatus_t udononnxSessionCompleteEvent(udononnxSession_t *session,
                                              cl_event completion,
                                              udononnxStatus_t result) {
  udononnxSessionState_t pending_state;
  if (!session || !completion)
    return UDONONNX_STATUS_INVALID_VALUE;
  if (session->state == UDONONNX_SESSION_DISPOSED)
    return UDONONNX_STATUS_STALE_HANDLE;
  pending_state = session->state;
  if ((pending_state != UDONONNX_SESSION_INITIALIZING &&
       pending_state != UDONONNX_SESSION_RUNNING) ||
      completion != session->active_event)
    return UDONONNX_STATUS_STALE_HANDLE;
  session->active_event = 0;
  session->last_status = result;
  session->last_error = udononnxStatusString(result);
  if (pending_state == UDONONNX_SESSION_INITIALIZING)
    session->state = result == UDONONNX_STATUS_SUCCESS
                         ? UDONONNX_SESSION_READY
                         : UDONONNX_SESSION_CREATED;
  else
    session->state = UDONONNX_SESSION_READY;
  return result;
}

udononnxStatus_t udononnxSessionDispose(udononnxSession_t *session) {
  udononnxStatus_t status;
  if (!session)
    return UDONONNX_STATUS_INVALID_VALUE;
  if (session->state == UDONONNX_SESSION_DISPOSED)
    return UDONONNX_STATUS_SUCCESS;
  if (session->state == UDONONNX_SESSION_INITIALIZING ||
      session->state == UDONONNX_SESSION_RUNNING)
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
  session->last_error = udononnxStatusString(UDONONNX_STATUS_SUCCESS);
  return UDONONNX_STATUS_SUCCESS;
}

udononnxStatus_t
udononnxSessionGetLastStatus(const udononnxSession_t *session) {
  return session ? session->last_status : UDONONNX_STATUS_INVALID_VALUE;
}

const char *udononnxSessionGetLastError(const udononnxSession_t *session) {
  return session && session->last_error
             ? session->last_error
             : udononnxStatusString(UDONONNX_STATUS_INVALID_VALUE);
}

const char *udononnxStatusString(udononnxStatus_t status) {
  if (status == UDONONNX_STATUS_SUCCESS)
    return "success";
  if (status == UDONONNX_STATUS_INVALID_VALUE)
    return "invalid value";
  if (status == UDONONNX_STATUS_TYPE_MISMATCH)
    return "type mismatch";
  if (status == UDONONNX_STATUS_STALE_HANDLE)
    return "stale handle";
  if (status == UDONONNX_STATUS_NOT_SUPPORTED)
    return "not supported";
  if (status == UDONONNX_STATUS_OPENCL_ERROR)
    return "OpenCL error";
  if (status == UDONONNX_STATUS_NOT_INITIALIZED)
    return "not initialized";
  if (status == UDONONNX_STATUS_BUSY)
    return "busy";
  if (status == UDONONNX_STATUS_DISPOSED)
    return "disposed";
  if (status == UDONONNX_STATUS_EXECUTION_FAILURE)
    return "execution failure";
  if (status == UDONONNX_STATUS_CONTEXT_MISMATCH)
    return "context mismatch";
  if (status == UDONONNX_STATUS_MISSING_BINDING)
    return "missing binding";
  return "unknown status";
}
