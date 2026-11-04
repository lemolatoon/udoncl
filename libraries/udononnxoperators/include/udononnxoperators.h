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

typedef uint32_t udononnxOperatorHandle_t;

typedef enum udononnxZeroPointFormat {
  UDONONNX_ZERO_POINT_NONE = 0,
  UDONONNX_ZERO_POINT_PACKED = 1,
  UDONONNX_ZERO_POINT_UNPACKED = 2
} udononnxZeroPointFormat_t;

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
udononnxStatus_t udononnxCreateOperatorHandle(udononnxOperatorHandle_t *handle);
udononnxStatus_t udononnxDestroyOperatorHandle(udononnxOperatorHandle_t handle);
/* Implements com.microsoft::MatMulNBits. B and packed zero points use the
 * ONNX Runtime block bitstream without requantization. A, scales, bias, and Y
 * use data_type and may be float32, float16, or bfloat16. */
udononnxStatus_t udononnxMatMulNBits(
    udononnxOperatorHandle_t handle, udonclDataType_t data_type, const void *a,
    int64_t m, int64_t k, const uint8_t *b, size_t b_bytes, const void *scales,
    size_t scale_count, const uint8_t *zero_points, size_t zero_point_bytes,
    udononnxZeroPointFormat_t zero_point_format, const int32_t *g_idx,
    size_t g_idx_count, const void *bias, size_t bias_count, int64_t n,
    int block_size, int bits, void *y);
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
