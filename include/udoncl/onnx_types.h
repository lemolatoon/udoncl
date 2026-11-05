#ifndef UDONCL_ONNX_TYPES_H
#define UDONCL_ONNX_TYPES_H

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
  UDONONNX_STATUS_EXECUTION_FAILURE = 9,
  UDONONNX_STATUS_CONTEXT_MISMATCH = 10,
  UDONONNX_STATUS_MISSING_BINDING = 11
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

#ifdef __cplusplus
}
#endif

#endif
