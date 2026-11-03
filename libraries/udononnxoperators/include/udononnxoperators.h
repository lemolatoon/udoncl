#ifndef UDONONNXOPERATORS_H
#define UDONONNXOPERATORS_H

#include <stddef.h>
#include <stdint.h>
#include <udoncl/device_memory.h>

#ifdef __cplusplus
extern "C" {
#endif

#define UDONONNX_MAX_TENSOR_RANK 8

typedef enum udononnxStatus {
  UDONONNX_STATUS_SUCCESS = 0,
  UDONONNX_STATUS_INVALID_VALUE = 1,
  UDONONNX_STATUS_TYPE_MISMATCH = 2,
  UDONONNX_STATUS_STALE_HANDLE = 3,
  UDONONNX_STATUS_NOT_SUPPORTED = 4,
  UDONONNX_STATUS_OPENCL_ERROR = 5,
  UDONONNX_STATUS_NOT_INITIALIZED = 6,
  UDONONNX_STATUS_BUSY = 7,
  UDONONNX_STATUS_DISPOSED = 8,
  UDONONNX_STATUS_EXECUTION_FAILURE = 9
} udononnxStatus_t;

/* A tensor descriptor is a non-owning logical view. The session or caller
 * owns the device allocation referenced by storage. */
typedef struct udononnxTensor {
  udonclBufferView_t storage;
  uint32_t rank;
  int64_t dimensions[UDONONNX_MAX_TENSOR_RANK];
} udononnxTensor_t;

/* A -1 dimension is dynamic and requires a positive value at the matching
 * position in dimension_bounds. Static dimensions ignore the bound. */
typedef struct udononnxTensorMetadata {
  const char *name;
  udonclDataType_t data_type;
  uint32_t rank;
  const int64_t *dimensions;
  const int64_t *dimension_bounds;
} udononnxTensorMetadata_t;

typedef struct udononnxRunToken {
  uint32_t session_generation;
  uint32_t run_generation;
} udononnxRunToken_t;

/* The struct must be zero-initialized before its first Initialize call.
 * Metadata and binding arrays are borrowed for the session lifetime. The
 * caller owns every buffer referenced by a binding. */
typedef struct udononnxSession {
  const udononnxTensorMetadata_t *inputs;
  const udononnxTensorMetadata_t *outputs;
  udononnxTensor_t *input_bindings;
  udononnxTensor_t *output_bindings;
  size_t input_count;
  size_t output_count;
  uint32_t generation;
  uint32_t run_generation;
  udononnxStatus_t last_status;
  int initialized;
  int busy;
  int disposed;
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
udononnxStatus_t udononnxSessionInitialize(
    udononnxSession_t *session, const udononnxTensorMetadata_t *inputs,
    size_t input_count, udononnxTensor_t *input_bindings,
    const udononnxTensorMetadata_t *outputs, size_t output_count,
    udononnxTensor_t *output_bindings);
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
udononnxStatus_t udononnxSessionStartRun(udononnxSession_t *session,
                                         udononnxRunToken_t *token);
udononnxStatus_t udononnxSessionCompleteRun(udononnxSession_t *session,
                                            udononnxRunToken_t token,
                                            udononnxStatus_t status);
udononnxStatus_t udononnxSessionDispose(udononnxSession_t *session);

#ifdef __cplusplus
}
#endif

#endif
