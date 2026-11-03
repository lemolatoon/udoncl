#ifndef UDONCL_DEVICE_MEMORY_H
#define UDONCL_DEVICE_MEMORY_H

#include <CL/cl.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum udonclStatus {
  UDONCL_STATUS_SUCCESS = 0,
  UDONCL_STATUS_INVALID_VALUE = 1,
  UDONCL_STATUS_OUT_OF_RESOURCES = 2,
  UDONCL_STATUS_STALE_HANDLE = 3,
  UDONCL_STATUS_OPENCL_ERROR = 4
} udonclStatus_t;

typedef enum udonclDataType {
  UDONCL_DATA_UINT8 = 1,
  UDONCL_DATA_INT32 = 2,
  UDONCL_DATA_INT64 = 3,
  UDONCL_DATA_FLOAT16 = 4,
  UDONCL_DATA_BFLOAT16 = 5,
  UDONCL_DATA_FLOAT32 = 6
} udonclDataType_t;

/* slot is one-based. A generation change invalidates every copied handle and
 * view referring to the previous allocation in that slot. */
typedef struct udonclBufferHandle {
  uint32_t slot;
  uint32_t generation;
} udonclBufferHandle_t;

/* Views never own storage. Offsets and sizes are bytes in the parent buffer. */
typedef struct udonclBufferView {
  udonclBufferHandle_t buffer;
  size_t byte_offset;
  size_t byte_length;
  udonclDataType_t data_type;
} udonclBufferView_t;

typedef struct udonclWorkspace {
  udonclBufferHandle_t buffer;
  size_t byte_offset;
  size_t byte_length;
} udonclWorkspace_t;

size_t udonclDataTypeSize(udonclDataType_t data_type);
udonclStatus_t udonclDeviceBufferCreate(cl_context context, cl_mem_flags flags,
                                        size_t bytes,
                                        udonclBufferHandle_t *buffer);
udonclStatus_t udonclDeviceBufferDestroy(udonclBufferHandle_t *buffer);
udonclStatus_t udonclDeviceBufferGetSize(udonclBufferHandle_t buffer,
                                         size_t *bytes);
udonclStatus_t udonclDeviceBufferGetRaw(udonclBufferHandle_t buffer,
                                        cl_mem *memory);
udonclStatus_t udonclDeviceBufferView(udonclBufferHandle_t buffer,
                                      size_t byte_offset, size_t elements,
                                      udonclDataType_t data_type,
                                      udonclBufferView_t *view);
udonclStatus_t udonclBufferViewResolve(udonclBufferView_t view, cl_mem *memory,
                                       size_t *byte_offset,
                                       size_t *byte_length);
udonclStatus_t udonclDeviceWorkspace(udonclBufferHandle_t buffer,
                                     size_t byte_offset, size_t byte_length,
                                     udonclWorkspace_t *workspace);
udonclStatus_t udonclWorkspaceResolve(udonclWorkspace_t workspace,
                                      cl_mem *memory, size_t *byte_offset,
                                      size_t *byte_length);

#ifdef __cplusplus
}
#endif

#endif
