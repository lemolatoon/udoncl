#include "udoncl/device_memory.h"

#include <limits.h>

#define UDONCL_DEVICE_BUFFER_CAPACITY 64

struct udoncl_device_buffer_entry {
  cl_mem memory;
  cl_context context;
  size_t bytes;
  uint32_t generation;
};

static struct udoncl_device_buffer_entry
    udoncl_device_buffers[UDONCL_DEVICE_BUFFER_CAPACITY];

static uint32_t udoncl_next_generation(uint32_t generation) {
  ++generation;
  return generation ? generation : 1u;
}

static udonclStatus_t
udoncl_device_buffer_find(udonclBufferHandle_t buffer,
                          struct udoncl_device_buffer_entry **entry) {
  struct udoncl_device_buffer_entry *candidate;
  if (!entry || !buffer.slot || buffer.slot > UDONCL_DEVICE_BUFFER_CAPACITY ||
      !buffer.generation)
    return UDONCL_STATUS_INVALID_VALUE;
  candidate = &udoncl_device_buffers[buffer.slot - 1u];
  if (!candidate->memory || candidate->generation != buffer.generation)
    return UDONCL_STATUS_STALE_HANDLE;
  *entry = candidate;
  return UDONCL_STATUS_SUCCESS;
}

static int udoncl_range_fits(size_t capacity, size_t offset, size_t bytes) {
  return offset <= capacity && bytes <= capacity - offset;
}

size_t udonclDataTypeSize(udonclDataType_t data_type) {
  if (data_type == UDONCL_DATA_UINT8)
    return 1;
  if (data_type == UDONCL_DATA_FLOAT16 || data_type == UDONCL_DATA_BFLOAT16)
    return 2;
  if (data_type == UDONCL_DATA_INT32 || data_type == UDONCL_DATA_FLOAT32)
    return 4;
  if (data_type == UDONCL_DATA_INT64)
    return 8;
  return 0;
}

udonclStatus_t udonclDeviceBufferCreate(cl_context context, cl_mem_flags flags,
                                        size_t bytes,
                                        udonclBufferHandle_t *buffer) {
  cl_mem memory;
  cl_int error = CL_SUCCESS;
  if (!buffer || !context || !bytes)
    return UDONCL_STATUS_INVALID_VALUE;
  buffer->slot = 0;
  buffer->generation = 0;
  memory = clCreateBuffer(context, flags, bytes, 0, &error);
  if (!memory)
    return error == CL_OUT_OF_RESOURCES || error == CL_OUT_OF_HOST_MEMORY
               ? UDONCL_STATUS_OUT_OF_RESOURCES
               : UDONCL_STATUS_OPENCL_ERROR;
  udonclStatus_t status =
      udonclDeviceBufferAdopt(context, memory, bytes, buffer);
  if (status != UDONCL_STATUS_SUCCESS)
    clReleaseMemObject(memory);
  return status;
}

udonclStatus_t udonclDeviceBufferAdopt(cl_context context, cl_mem memory,
                                       size_t bytes,
                                       udonclBufferHandle_t *buffer) {
  uint32_t index;
  if (!buffer || !context || !memory || !bytes)
    return UDONCL_STATUS_INVALID_VALUE;
  buffer->slot = 0;
  buffer->generation = 0;
  for (index = 0; index < UDONCL_DEVICE_BUFFER_CAPACITY; ++index) {
    struct udoncl_device_buffer_entry *entry = &udoncl_device_buffers[index];
    if (entry->memory)
      continue;
    entry->memory = memory;
    entry->bytes = bytes;
    entry->context = context;
    entry->generation = udoncl_next_generation(entry->generation);
    buffer->slot = index + 1u;
    buffer->generation = entry->generation;
    return UDONCL_STATUS_SUCCESS;
  }
  return UDONCL_STATUS_OUT_OF_RESOURCES;
}

udonclStatus_t udonclDeviceBufferDestroy(udonclBufferHandle_t *buffer) {
  struct udoncl_device_buffer_entry *entry;
  udonclStatus_t status;
  cl_int error;
  if (!buffer)
    return UDONCL_STATUS_INVALID_VALUE;
  status = udoncl_device_buffer_find(*buffer, &entry);
  if (status != UDONCL_STATUS_SUCCESS)
    return status;
  error = clReleaseMemObject(entry->memory);
  if (error != CL_SUCCESS)
    return UDONCL_STATUS_OPENCL_ERROR;
  entry->memory = 0;
  entry->context = 0;
  entry->bytes = 0;
  buffer->slot = 0;
  buffer->generation = 0;
  return UDONCL_STATUS_SUCCESS;
}

udonclStatus_t udonclDeviceBufferGetSize(udonclBufferHandle_t buffer,
                                         size_t *bytes) {
  struct udoncl_device_buffer_entry *entry;
  udonclStatus_t status;
  if (!bytes)
    return UDONCL_STATUS_INVALID_VALUE;
  status = udoncl_device_buffer_find(buffer, &entry);
  if (status == UDONCL_STATUS_SUCCESS)
    *bytes = entry->bytes;
  return status;
}

udonclStatus_t udonclDeviceBufferGetRaw(udonclBufferHandle_t buffer,
                                        cl_mem *memory) {
  struct udoncl_device_buffer_entry *entry;
  udonclStatus_t status;
  if (!memory)
    return UDONCL_STATUS_INVALID_VALUE;
  status = udoncl_device_buffer_find(buffer, &entry);
  if (status == UDONCL_STATUS_SUCCESS)
    *memory = entry->memory;
  return status;
}

udonclStatus_t udonclDeviceBufferGetContext(udonclBufferHandle_t buffer,
                                            cl_context *context) {
  struct udoncl_device_buffer_entry *entry;
  udonclStatus_t status;
  if (!context)
    return UDONCL_STATUS_INVALID_VALUE;
  status = udoncl_device_buffer_find(buffer, &entry);
  if (status == UDONCL_STATUS_SUCCESS)
    *context = entry->context;
  return status;
}

udonclStatus_t udonclDeviceBufferView(udonclBufferHandle_t buffer,
                                      size_t byte_offset, size_t elements,
                                      udonclDataType_t data_type,
                                      udonclBufferView_t *view) {
  struct udoncl_device_buffer_entry *entry;
  udonclStatus_t status;
  size_t element_bytes = udonclDataTypeSize(data_type);
  size_t byte_length;
  if (!view || !element_bytes ||
      (uint64_t)elements * element_bytes > SIZE_MAX ||
      (byte_offset & (element_bytes - 1u)))
    return UDONCL_STATUS_INVALID_VALUE;
  byte_length = elements * element_bytes;
  status = udoncl_device_buffer_find(buffer, &entry);
  if (status != UDONCL_STATUS_SUCCESS)
    return status;
  if (!udoncl_range_fits(entry->bytes, byte_offset, byte_length))
    return UDONCL_STATUS_INVALID_VALUE;
  view->buffer = buffer;
  view->byte_offset = byte_offset;
  view->byte_length = byte_length;
  view->data_type = data_type;
  return UDONCL_STATUS_SUCCESS;
}

udonclStatus_t udonclBufferViewResolve(udonclBufferView_t view, cl_mem *memory,
                                       size_t *byte_offset,
                                       size_t *byte_length) {
  struct udoncl_device_buffer_entry *entry;
  udonclStatus_t status;
  size_t element_bytes = udonclDataTypeSize(view.data_type);
  if (!memory || !byte_offset || !byte_length || !element_bytes ||
      (view.byte_offset & (element_bytes - 1u)) ||
      (view.byte_length & (element_bytes - 1u)))
    return UDONCL_STATUS_INVALID_VALUE;
  status = udoncl_device_buffer_find(view.buffer, &entry);
  if (status != UDONCL_STATUS_SUCCESS)
    return status;
  if (!udoncl_range_fits(entry->bytes, view.byte_offset, view.byte_length))
    return UDONCL_STATUS_INVALID_VALUE;
  *memory = entry->memory;
  *byte_offset = view.byte_offset;
  *byte_length = view.byte_length;
  return UDONCL_STATUS_SUCCESS;
}

udonclStatus_t udonclDeviceWorkspace(udonclBufferHandle_t buffer,
                                     size_t byte_offset, size_t byte_length,
                                     udonclWorkspace_t *workspace) {
  struct udoncl_device_buffer_entry *entry;
  udonclStatus_t status;
  if (!workspace)
    return UDONCL_STATUS_INVALID_VALUE;
  status = udoncl_device_buffer_find(buffer, &entry);
  if (status != UDONCL_STATUS_SUCCESS)
    return status;
  if (!udoncl_range_fits(entry->bytes, byte_offset, byte_length))
    return UDONCL_STATUS_INVALID_VALUE;
  workspace->buffer = buffer;
  workspace->byte_offset = byte_offset;
  workspace->byte_length = byte_length;
  return UDONCL_STATUS_SUCCESS;
}

udonclStatus_t udonclWorkspaceResolve(udonclWorkspace_t workspace,
                                      cl_mem *memory, size_t *byte_offset,
                                      size_t *byte_length) {
  struct udoncl_device_buffer_entry *entry;
  udonclStatus_t status;
  if (!memory || !byte_offset || !byte_length)
    return UDONCL_STATUS_INVALID_VALUE;
  status = udoncl_device_buffer_find(workspace.buffer, &entry);
  if (status != UDONCL_STATUS_SUCCESS)
    return status;
  if (!udoncl_range_fits(entry->bytes, workspace.byte_offset,
                         workspace.byte_length))
    return UDONCL_STATUS_INVALID_VALUE;
  *memory = entry->memory;
  *byte_offset = workspace.byte_offset;
  *byte_length = workspace.byte_length;
  return UDONCL_STATUS_SUCCESS;
}
