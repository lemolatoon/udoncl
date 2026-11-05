#ifndef UDONONNXRUNTIME_H
#define UDONONNXRUNTIME_H

#include <CL/cl.h>
#include <stddef.h>
#include <stdint.h>
#include <udoncl/onnx_types.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct udononnxModel {
  const char *name;
  uint32_t graph_id;
  const udononnxTensorMetadata_t *inputs;
  size_t input_count;
  const udononnxTensorMetadata_t *outputs;
  size_t output_count;
} udononnxModel_t;

typedef enum udononnxSessionState {
  UDONONNX_SESSION_EMPTY = 0,
  UDONONNX_SESSION_CREATED = 1,
  UDONONNX_SESSION_INITIALIZING = 2,
  UDONONNX_SESSION_READY = 3,
  UDONONNX_SESSION_RUNNING = 4,
  UDONONNX_SESSION_DISPOSED = 5
} udononnxSessionState_t;

/* The model metadata and binding arrays are borrowed for the session
 * lifetime. Every bound device allocation remains owned by the caller. */
typedef struct udononnxSession {
  const udononnxModel_t *model;
  cl_context context;
  udononnxTensor_t *input_bindings;
  udononnxTensor_t *output_bindings;
  uint32_t generation;
  uint32_t operation_generation;
  udononnxStatus_t last_status;
  const char *last_error;
  cl_event active_event;
  udononnxSessionState_t state;
} udononnxSession_t;

udononnxStatus_t udononnxCreateTensor(udonclBufferView_t storage, uint32_t rank,
                                      const int64_t *dimensions,
                                      udononnxTensor_t *tensor);
udononnxStatus_t udononnxGetTensorElementCount(udononnxTensor_t tensor,
                                               size_t *elements);
udononnxStatus_t udononnxGetTensorDimension(udononnxTensor_t tensor,
                                            uint32_t axis, int64_t *dimension);
udononnxStatus_t udononnxGetTensorStorage(udononnxTensor_t tensor,
                                          udonclBufferView_t *storage);
udononnxStatus_t udononnxValidateTensor(udononnxTensor_t tensor);

udononnxStatus_t udononnxFindModel(const udononnxModel_t *models,
                                   size_t model_count, const char *name,
                                   const udononnxModel_t **model);
size_t udononnxGeneratedGetModelCount(void);
udononnxStatus_t udononnxGeneratedGetModel(size_t index,
                                           const udononnxModel_t **model);
size_t udononnxGetModelCount(void);
udononnxStatus_t udononnxGetModel(size_t index, const udononnxModel_t **model);
udononnxStatus_t udononnxFindGeneratedModel(const char *name,
                                            const udononnxModel_t **model);
udononnxStatus_t udononnxCreateSession(udononnxSession_t *session,
                                       const udononnxModel_t *model,
                                       cl_context context,
                                       udononnxTensor_t *input_bindings,
                                       udononnxTensor_t *output_bindings);
size_t udononnxSessionGetInputCount(const udononnxSession_t *session);
size_t udononnxSessionGetOutputCount(const udononnxSession_t *session);
const udononnxTensorMetadata_t *
udononnxSessionGetInputMetadata(const udononnxSession_t *session, size_t index);
const udononnxTensorMetadata_t *
udononnxSessionGetOutputMetadata(const udononnxSession_t *session,
                                 size_t index);
int udononnxSessionFindInput(const udononnxSession_t *session,
                             const char *name);
int udononnxSessionFindOutput(const udononnxSession_t *session,
                              const char *name);
udononnxStatus_t udononnxSessionBindInput(udononnxSession_t *session,
                                          size_t index,
                                          udononnxTensor_t tensor);
udononnxStatus_t udononnxSessionBindOutput(udononnxSession_t *session,
                                           size_t index,
                                           udononnxTensor_t tensor);
udononnxStatus_t udononnxSessionResolveInput(const udononnxSession_t *session,
                                             size_t index, cl_mem *memory,
                                             size_t *byte_offset,
                                             size_t *byte_length);
udononnxStatus_t udononnxSessionResolveOutput(const udononnxSession_t *session,
                                              size_t index, cl_mem *memory,
                                              size_t *byte_offset,
                                              size_t *byte_length);

/* These generated, whole-program-visible dispatchers are supplied by the
 * linked model host. They dispatch graph_id with direct calls; they are not a
 * public model-specific API or an external execution plan. */
udononnxStatus_t udononnxGeneratedEnqueueInitialize(
    uint32_t graph_id, udononnxSession_t *session, cl_command_queue queue,
    cl_uint wait_count, const cl_event *wait_list, cl_event *completion);
udononnxStatus_t
udononnxGeneratedEnqueueRun(uint32_t graph_id, udononnxSession_t *session,
                            cl_command_queue queue, cl_uint wait_count,
                            const cl_event *wait_list, cl_event *completion);

udononnxStatus_t udononnxEnqueueInitialize(udononnxSession_t *session,
                                           cl_command_queue queue,
                                           cl_uint wait_count,
                                           const cl_event *wait_list,
                                           cl_event *completion);
udononnxStatus_t udononnxEnqueueRun(udononnxSession_t *session,
                                    cl_command_queue queue, cl_uint wait_count,
                                    const cl_event *wait_list,
                                    cl_event *completion);
/* Called by the Host continuation that observes completion. It never releases
 * the caller-owned event. */
udononnxStatus_t udononnxSessionCompleteEvent(udononnxSession_t *session,
                                              cl_event completion,
                                              udononnxStatus_t result);
udononnxStatus_t udononnxSessionDispose(udononnxSession_t *session);
udononnxStatus_t udononnxSessionGetLastStatus(const udononnxSession_t *session);
const char *udononnxSessionGetLastError(const udononnxSession_t *session);
const char *udononnxStatusString(udononnxStatus_t status);

#ifdef __cplusplus
}
#endif

#endif
