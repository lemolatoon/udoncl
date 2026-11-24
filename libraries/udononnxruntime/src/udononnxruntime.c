#include "udononnxruntime_internal.h"

static int udononnxStringEqual(const char *left, const char *right) {
  if (!left || !right)
    return 0;
  while (*left && *left == *right) {
    ++left;
    ++right;
  }
  return *left == *right;
}

/* The index of the graph tensor `name` among `count`, or -1. */
static int udononnxFindTensor(const udononnxTensorMetadata_t *metadata,
                              size_t count, const char *name) {
  size_t index;
  if (!metadata || !name)
    return -1;
  for (index = 0; index < count; ++index)
    if (udononnxStringEqual(metadata[index].name, name))
      return (int)index;
  return -1;
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

/* The UdonONNX session (udononnxruntime.h). The model is compiled into the
 * Host, so a Host has one session object; Create hands it out while it is
 * not live. The buffers it binds are kept by the generated binding
 * (udononnxGeneratedBindMemory), which the generated run reads. */
struct udononnxSession {
  const udononnxModel_t *model;
  cl_context context;
  cl_command_queue queue;
  uint32_t live;
};

static udononnxSession_t udononnx_session;

/* NOT_INITIALIZED for a session that is not live (or none), INVALID_VALUE
 * for a pointer that is not the session of this Host. */
static udononnxStatus_t udononnxCheckSession(const udononnxSession_t *session) {
  if (!session)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  if (session != &udononnx_session)
    return UDONONNX_STATUS_INVALID_VALUE;
  return session->live ? UDONONNX_STATUS_SUCCESS
                       : UDONONNX_STATUS_NOT_INITIALIZED;
}

udononnxStatus_t udononnxCreateSession(cl_context context,
                                       cl_command_queue queue,
                                       udononnxSession_t **session) {
  const udononnxModel_t *model = 0;
  cl_context queue_context = 0;
  udononnxStatus_t status;
  if (!session)
    return UDONONNX_STATUS_INVALID_VALUE;
  *session = 0;
  if (!context || !queue)
    return UDONONNX_STATUS_INVALID_VALUE;
  if (udononnx_session.live)
    return UDONONNX_STATUS_INVALID_STATE;
  if (clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(queue_context),
                            &queue_context, 0) != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  if (queue_context != context)
    return UDONONNX_STATUS_CONTEXT_MISMATCH;
  status = udononnxGetModel(0, &model);
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  status = udononnxGeneratedCreate(model->graph_id, 0, queue);
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  udononnx_session.model = model;
  udononnx_session.context = context;
  udononnx_session.queue = queue;
  udononnx_session.live = 1;
  *session = &udononnx_session;
  return UDONONNX_STATUS_SUCCESS;
}

static udononnxStatus_t udononnxBindAt(udononnxSession_t *session,
                                       uint32_t output, int index,
                                       cl_mem memory, int rank,
                                       const int64_t *dims,
                                       const int64_t *buffer_dims) {
  const udononnxTensorMetadata_t *metadata;
  size_t count;
  uint32_t axis;
  udononnxStatus_t status = udononnxCheckSession(session);
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  count = output ? session->model->output_count : session->model->input_count;
  if (index < 0 || (size_t)index >= count || !memory)
    return UDONONNX_STATUS_INVALID_VALUE;
  metadata =
      output ? &session->model->outputs[index] : &session->model->inputs[index];
  if (rank < 0 || (uint32_t)rank != metadata->rank || (rank && !dims))
    return UDONONNX_STATUS_INVALID_VALUE;
  for (axis = 0; axis < metadata->rank; ++axis) {
    /* A dimension the model leaves dynamic is bound with its extent once
     * the run supports it. */
    if (metadata->dimensions[axis] < 0)
      return UDONONNX_STATUS_NOT_SUPPORTED;
    if (dims[axis] != metadata->dimensions[axis])
      return UDONONNX_STATUS_INVALID_VALUE;
  }
  /* A buffer larger than the tensor is bound with the extent it has once
   * the run supports it; buffer_dims equal to dims is the tensor itself. */
  if (buffer_dims)
    for (axis = 0; axis < metadata->rank; ++axis)
      if (buffer_dims[axis] != dims[axis])
        return UDONONNX_STATUS_NOT_SUPPORTED;
  return udononnxGeneratedBindMemory(session->model->graph_id, output,
                                     (uint32_t)index, memory);
}

udononnxStatus_t udononnxBindInputAt(udononnxSession_t *session, int index,
                                     cl_mem memory, int rank,
                                     const int64_t *dims,
                                     const int64_t *buffer_dims) {
  return udononnxBindAt(session, 0, index, memory, rank, dims, buffer_dims);
}

udononnxStatus_t udononnxBindOutputAt(udononnxSession_t *session, int index,
                                      cl_mem memory, int rank,
                                      const int64_t *dims,
                                      const int64_t *buffer_dims) {
  return udononnxBindAt(session, 1, index, memory, rank, dims, buffer_dims);
}

udononnxStatus_t udononnxBindInput(udononnxSession_t *session, const char *name,
                                   cl_mem memory, int rank, const int64_t *dims,
                                   const int64_t *buffer_dims) {
  int index;
  udononnxStatus_t status = udononnxCheckSession(session);
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  index = udononnxFindTensor(session->model->inputs,
                             session->model->input_count, name);
  if (index < 0)
    return UDONONNX_STATUS_INVALID_VALUE;
  return udononnxBindInputAt(session, index, memory, rank, dims, buffer_dims);
}

udononnxStatus_t udononnxBindOutput(udononnxSession_t *session,
                                    const char *name, cl_mem memory, int rank,
                                    const int64_t *dims,
                                    const int64_t *buffer_dims) {
  int index;
  udononnxStatus_t status = udononnxCheckSession(session);
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  index = udononnxFindTensor(session->model->outputs,
                             session->model->output_count, name);
  if (index < 0)
    return UDONONNX_STATUS_INVALID_VALUE;
  return udononnxBindOutputAt(session, index, memory, rank, dims, buffer_dims);
}

udononnxStatus_t udononnxRun(udononnxSession_t *session) {
  cl_event completion = 0;
  udononnxStatus_t status = udononnxCheckSession(session);
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  /* The run is on the in-order queue: nothing waits for its completion
   * marker, which is released right away. */
  status = udononnxGeneratedRun(session->model->graph_id, session->queue,
                                &completion);
  if (completion && clReleaseEvent(completion) != CL_SUCCESS &&
      status == UDONONNX_STATUS_SUCCESS)
    status = UDONONNX_STATUS_OPENCL_ERROR;
  return status;
}

udononnxStatus_t udononnxReleaseSession(udononnxSession_t *session) {
  udononnxStatus_t status = udononnxCheckSession(session);
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  status = udononnxGeneratedDispose(session->model->graph_id, 0);
  session->model = 0;
  session->context = 0;
  session->queue = 0;
  session->live = 0;
  return status;
}
