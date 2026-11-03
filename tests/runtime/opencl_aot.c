#include <CL/cl.h>
#include <stddef.h>

#define EXPECT(condition)                                                      \
  do {                                                                         \
    if (!(condition))                                                          \
      return __LINE__;                                                         \
  } while (0)

static cl_uint allocated_logical_id;
static size_t allocated_bytes;
static cl_uint uploaded_token;
static cl_uint released_token;

cl_uint __vrc_opencl_buffer_alloc(cl_uint logical_id, size_t bytes) {
  allocated_logical_id = logical_id;
  allocated_bytes = bytes;
  return logical_id + 100;
}

cl_int __vrc_opencl_buffer_release(cl_uint physical_token) {
  released_token = physical_token;
  return CL_SUCCESS;
}

cl_int __vrc_opencl_buffer_upload(cl_uint physical_token, size_t offset,
                                  size_t bytes, const void *source) {
  (void)offset;
  (void)bytes;
  (void)source;
  uploaded_token = physical_token;
  return 7;
}

cl_int __vrc_opencl_buffer_readback(cl_uint physical_token, size_t offset,
                                    size_t bytes, void *destination) {
  (void)physical_token;
  (void)offset;
  (void)bytes;
  (void)destination;
  return 8;
}

cl_int __vrc_opencl_dispatch(cl_uint kernel_id,
                             const unsigned char *arguments,
                             size_t argument_bytes, cl_uint work_dim,
                             const size_t *global_offset,
                             const size_t *global_size,
                             const size_t *local_size) {
  (void)kernel_id;
  (void)arguments;
  (void)argument_bytes;
  (void)work_dim;
  (void)global_offset;
  (void)global_size;
  (void)local_size;
  return 9;
}

cl_int __vrc_opencl_await(cl_int event) { return event; }

int main(void) {
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
  cl_mem first, second;
  cl_int error;
  int source = 42;

  EXPECT(clGetPlatformIDs(1, &platform, 0) == CL_SUCCESS);
  EXPECT(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, 0) ==
         CL_SUCCESS);
  context = clCreateContext(0, 1, &device, 0, 0, &error);
  EXPECT(context && error == CL_SUCCESS);
  queue = clCreateCommandQueue(context, device, 0, &error);
  EXPECT(queue && error == CL_SUCCESS);

  first = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(source), 0, &error);
  EXPECT(first && error == CL_SUCCESS);
  EXPECT(allocated_logical_id == 1 && allocated_bytes == sizeof(source));
  EXPECT(clEnqueueWriteBuffer(queue, first, CL_FALSE, 0, sizeof(source),
                              &source, 0, 0, 0) == CL_SUCCESS);
  EXPECT(uploaded_token == 101);
  EXPECT(clReleaseMemObject(first) == CL_SUCCESS);
  EXPECT(released_token == 101);

  /* Reusing the logical table slot must request a new physical token rather
   * than retaining the released backing-resource identity. */
  second =
      clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(source), 0, &error);
  EXPECT(second && error == CL_SUCCESS);
  EXPECT(allocated_logical_id == 1);
  EXPECT(clReleaseMemObject(second) == CL_SUCCESS);
  EXPECT(released_token == 101);
  return 0;
}
