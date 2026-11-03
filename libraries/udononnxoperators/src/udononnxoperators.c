#include "udononnxoperators.h"

#include <limits.h>

static udononnxStatus_t udononnxElementCount(uint32_t rank,
                                             const int64_t *dimensions,
                                             size_t *elements) {
  uint32_t axis;
  size_t count = 1;
  if (!elements || rank > UDONONNX_MAX_TENSOR_RANK || (rank && !dimensions))
    return UDONONNX_STATUS_INVALID_VALUE;
  for (axis = 0; axis < rank; ++axis) {
    size_t dimension;
    if (dimensions[axis] <= 0 || (uint64_t)dimensions[axis] > SIZE_MAX)
      return UDONONNX_STATUS_INVALID_VALUE;
    dimension = (size_t)dimensions[axis];
    if (count > SIZE_MAX / dimension)
      return UDONONNX_STATUS_INVALID_VALUE;
    count *= dimension;
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
  if (!element_bytes || elements > SIZE_MAX / element_bytes ||
      storage.byte_length != elements * element_bytes)
    return UDONONNX_STATUS_TYPE_MISMATCH;
  tensor->storage = storage;
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
    *storage = tensor.storage;
  return status;
}

udononnxStatus_t udononnxValidateTensor(udononnxTensor_t tensor) {
  cl_mem memory;
  size_t byte_offset, byte_length, elements, element_bytes;
  udonclStatus_t storage_status;
  udononnxStatus_t status =
      udononnxElementCount(tensor.rank, tensor.dimensions, &elements);
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  element_bytes = udonclDataTypeSize(tensor.storage.data_type);
  if (!element_bytes || elements > SIZE_MAX / element_bytes ||
      tensor.storage.byte_length != elements * element_bytes)
    return UDONONNX_STATUS_TYPE_MISMATCH;
  storage_status = udonclBufferViewResolve(tensor.storage, &memory,
                                           &byte_offset, &byte_length);
  if (storage_status == UDONCL_STATUS_STALE_HANDLE)
    return UDONONNX_STATUS_STALE_HANDLE;
  if (storage_status != UDONCL_STATUS_SUCCESS)
    return UDONONNX_STATUS_INVALID_VALUE;
  return UDONONNX_STATUS_SUCCESS;
}

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
                        udononnxTensor_t tensor) {
  uint32_t axis;
  udononnxStatus_t status = udononnxValidateMetadata(metadata);
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  status = udononnxValidateTensor(tensor);
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  if (tensor.storage.data_type != metadata->data_type ||
      tensor.rank != metadata->rank)
    return UDONONNX_STATUS_TYPE_MISMATCH;
  for (axis = 0; axis < tensor.rank; ++axis) {
    if (metadata->dimensions[axis] > 0 &&
        tensor.dimensions[axis] != metadata->dimensions[axis])
      return UDONONNX_STATUS_TYPE_MISMATCH;
    if (metadata->dimensions[axis] == -1 &&
        tensor.dimensions[axis] > metadata->dimension_bounds[axis])
      return UDONONNX_STATUS_INVALID_VALUE;
  }
  return UDONONNX_STATUS_SUCCESS;
}

static udononnxStatus_t udononnxSessionReady(udononnxSession_t *session) {
  if (!session)
    return UDONONNX_STATUS_INVALID_VALUE;
  if (session->disposed)
    return UDONONNX_STATUS_DISPOSED;
  if (!session->initialized)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  if (session->busy)
    return UDONONNX_STATUS_BUSY;
  return UDONONNX_STATUS_SUCCESS;
}

udononnxStatus_t udononnxSessionInitialize(
    udononnxSession_t *session, const udononnxTensorMetadata_t *inputs,
    size_t input_count, udononnxTensor_t *input_bindings,
    const udononnxTensorMetadata_t *outputs, size_t output_count,
    udononnxTensor_t *output_bindings) {
  size_t index;
  uint32_t generation;
  if (!session || input_count > INT_MAX || output_count > INT_MAX ||
      (input_count && (!inputs || !input_bindings)) ||
      (output_count && (!outputs || !output_bindings)))
    return UDONONNX_STATUS_INVALID_VALUE;
  for (index = 0; index < input_count; ++index)
    if (udononnxValidateMetadata(&inputs[index]) != UDONONNX_STATUS_SUCCESS)
      return UDONONNX_STATUS_INVALID_VALUE;
  for (index = 0; index < output_count; ++index)
    if (udononnxValidateMetadata(&outputs[index]) != UDONONNX_STATUS_SUCCESS)
      return UDONONNX_STATUS_INVALID_VALUE;
  generation = udononnxNextGeneration(session->generation);
  session->inputs = inputs;
  session->outputs = outputs;
  session->input_bindings = input_bindings;
  session->output_bindings = output_bindings;
  session->input_count = input_count;
  session->output_count = output_count;
  session->generation = generation;
  session->run_generation = 0;
  session->last_status = UDONONNX_STATUS_SUCCESS;
  session->initialized = 1;
  session->busy = 0;
  session->disposed = 0;
  for (index = 0; index < input_count; ++index)
    input_bindings[index].storage.buffer.slot = 0;
  for (index = 0; index < output_count; ++index)
    output_bindings[index].storage.buffer.slot = 0;
  return UDONONNX_STATUS_SUCCESS;
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
  return session && session->initialized && !session->disposed
             ? udononnxSessionFind(session->inputs, session->input_count, name)
             : -1;
}

int udononnxSessionFindOutput(const udononnxSession_t *session,
                              const char *name) {
  return session && session->initialized && !session->disposed
             ? udononnxSessionFind(session->outputs, session->output_count,
                                   name)
             : -1;
}

static udononnxStatus_t
udononnxSessionBind(udononnxSession_t *session, size_t index,
                    udononnxTensor_t tensor,
                    const udononnxTensorMetadata_t *metadata, size_t count,
                    udononnxTensor_t *bindings) {
  udononnxStatus_t status = udononnxSessionReady(session);
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  if (index >= count)
    return UDONONNX_STATUS_INVALID_VALUE;
  status = udononnxValidateBinding(&metadata[index], tensor);
  if (status == UDONONNX_STATUS_SUCCESS)
    bindings[index] = tensor;
  return status;
}

udononnxStatus_t udononnxSessionBindInput(udononnxSession_t *session,
                                          size_t index,
                                          udononnxTensor_t tensor) {
  return session ? udononnxSessionBind(session, index, tensor, session->inputs,
                                       session->input_count,
                                       session->input_bindings)
                 : UDONONNX_STATUS_INVALID_VALUE;
}

udononnxStatus_t udononnxSessionBindOutput(udononnxSession_t *session,
                                           size_t index,
                                           udononnxTensor_t tensor) {
  return session ? udononnxSessionBind(session, index, tensor, session->outputs,
                                       session->output_count,
                                       session->output_bindings)
                 : UDONONNX_STATUS_INVALID_VALUE;
}

udononnxStatus_t udononnxSessionStartRun(udononnxSession_t *session,
                                         udononnxRunToken_t *token) {
  size_t index;
  udononnxStatus_t status = udononnxSessionReady(session);
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  if (!token)
    return UDONONNX_STATUS_INVALID_VALUE;
  for (index = 0; index < session->input_count; ++index) {
    status = udononnxValidateBinding(&session->inputs[index],
                                     session->input_bindings[index]);
    if (status != UDONONNX_STATUS_SUCCESS)
      return status;
  }
  for (index = 0; index < session->output_count; ++index) {
    status = udononnxValidateBinding(&session->outputs[index],
                                     session->output_bindings[index]);
    if (status != UDONONNX_STATUS_SUCCESS)
      return status;
  }
  session->run_generation = udononnxNextGeneration(session->run_generation);
  session->busy = 1;
  session->last_status = UDONONNX_STATUS_BUSY;
  token->session_generation = session->generation;
  token->run_generation = session->run_generation;
  return UDONONNX_STATUS_SUCCESS;
}

udononnxStatus_t udononnxSessionCompleteRun(udononnxSession_t *session,
                                            udononnxRunToken_t token,
                                            udononnxStatus_t status) {
  if (!session)
    return UDONONNX_STATUS_INVALID_VALUE;
  if (session->disposed || token.session_generation != session->generation ||
      token.run_generation != session->run_generation)
    return UDONONNX_STATUS_STALE_HANDLE;
  if (!session->busy)
    return UDONONNX_STATUS_INVALID_VALUE;
  session->busy = 0;
  session->last_status = status;
  return status;
}

udononnxStatus_t udononnxSessionDispose(udononnxSession_t *session) {
  if (!session)
    return UDONONNX_STATUS_INVALID_VALUE;
  if (session->disposed)
    return UDONONNX_STATUS_SUCCESS;
  if (session->busy)
    return UDONONNX_STATUS_BUSY;
  session->generation = udononnxNextGeneration(session->generation);
  session->initialized = 0;
  session->disposed = 1;
  return UDONONNX_STATUS_SUCCESS;
}
