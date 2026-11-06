#include "udondnn_device_source.h"
#include <CL/cl.h>
#include <VRC/Runtime/OpenCL/Backend.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <udondnn.h>

#define EXPECT(condition)                                                      \
  do {                                                                         \
    if (!(condition))                                                          \
      return __LINE__;                                                         \
  } while (0)

size_t udoncl_cl_buffer_limit(void) { return 1024 * 1024; }

int udoncl_cl_gather_profile(const char *name,
                             struct udoncl_cl_gather_profile *profile) {
  (void)name;
  (void)profile;
  return 0;
}

int udoncl_cl_draw_gather(uint32_t kernel, const unsigned char *const *inputs,
                          size_t input_count, size_t bytes,
                          const unsigned char *scalars, size_t scalar_bytes,
                          unsigned char *output) {
  (void)kernel;
  (void)inputs;
  (void)input_count;
  (void)bytes;
  (void)scalars;
  (void)scalar_bytes;
  (void)output;
  return 0;
}

int udoncl_cl_source_matches(const char *source, size_t length) {
  return source && length == strlen(UDONDNN_DEVICE_SOURCE) &&
         !memcmp(source, UDONDNN_DEVICE_SOURCE, length);
}

int udoncl_cl_kernel_matches(const char *name) {
  (void)name;
  return 0;
}

int udoncl_cl_draw(uint32_t buffer, uint32_t generation,
                   const unsigned char scalar[4], unsigned char result[4]) {
  (void)buffer;
  (void)generation;
  (void)scalar;
  (void)result;
  return 0;
}

int main(void) {
  udondnnHandle_t handle = 0;
  cl_platform_id platform = 0;
  cl_device_id device = 0;
  cl_context initial_context = 0, foreign_context = 0;
  cl_command_queue initial = 0, replacement = 0, foreign = 0, current = 0;
  cl_int error = CL_SUCCESS;

  EXPECT(udondnnCreate(&handle) == UDONDNN_STATUS_SUCCESS);
  EXPECT(udondnnGetStream(handle, &initial) == UDONDNN_STATUS_SUCCESS);
  EXPECT(initial != 0);
  EXPECT(clGetCommandQueueInfo(initial, CL_QUEUE_CONTEXT,
                               sizeof(initial_context), &initial_context,
                               0) == CL_SUCCESS);
  EXPECT(clGetPlatformIDs(1, &platform, 0) == CL_SUCCESS);
  EXPECT(clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, 0) ==
         CL_SUCCESS);

  replacement = clCreateCommandQueue(initial_context, device, 0, &error);
  EXPECT(replacement && error == CL_SUCCESS);
  EXPECT(udondnnSetStream(handle, replacement) == UDONDNN_STATUS_SUCCESS);
  EXPECT(udondnnGetStream(handle, &current) == UDONDNN_STATUS_SUCCESS);
  EXPECT(current == replacement);
  EXPECT(udondnnSetStream(handle, 0) == UDONDNN_STATUS_BAD_PARAM);
  EXPECT(udondnnGetStream(handle, &current) == UDONDNN_STATUS_SUCCESS);
  EXPECT(current == replacement);

  foreign_context = clCreateContext(0, 1, &device, 0, 0, &error);
  EXPECT(foreign_context && error == CL_SUCCESS);
  foreign = clCreateCommandQueue(foreign_context, device, 0, &error);
  EXPECT(foreign && error == CL_SUCCESS);
  EXPECT(udondnnSetStream(handle, foreign) == UDONDNN_STATUS_SUCCESS);
  EXPECT(udondnnGetStream(handle, &current) == UDONDNN_STATUS_SUCCESS);
  EXPECT(current == foreign);

  EXPECT(clReleaseCommandQueue(replacement) == CL_SUCCESS);
  EXPECT(clReleaseCommandQueue(foreign) == CL_SUCCESS);
  EXPECT(clReleaseContext(foreign_context) == CL_SUCCESS);
  EXPECT(udondnnDestroy(handle) == UDONDNN_STATUS_SUCCESS);
  EXPECT(udoncl_cl_live_objects() == 0);
  return 0;
}
