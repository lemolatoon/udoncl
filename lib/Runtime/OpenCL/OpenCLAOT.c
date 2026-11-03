/* OpenCL 1.2 Host subset for the Udon AOT backend.
 *
 * This translation unit deliberately has no hosted-libc dependency. Logical
 * OpenCL objects live in fixed per-library-instance storage; only __vrc_*
 * functions cross into the compiler/Unity backend. The native test runtime
 * remains OpenCL.c and implements the same public OpenCL calls independently.
 */
#include <CL/cl.h>

#define UDONCL_AOT_BUFFER_CAPACITY 8
#define UDONCL_AOT_KERNEL_CAPACITY 16
#define UDONCL_AOT_ARGUMENT_BYTES 128

extern cl_int __vrc_opencl_buffer_alloc(cl_uint id, size_t bytes);
extern cl_int __vrc_opencl_buffer_release(cl_uint id);
extern cl_int __vrc_opencl_buffer_upload(cl_uint id, size_t offset,
                                         size_t bytes, const void *source);
extern cl_int __vrc_opencl_buffer_readback(cl_uint id, size_t offset,
                                           size_t bytes, void *destination);
extern cl_int __vrc_opencl_dispatch(cl_uint kernel_id,
                                    const unsigned char *arguments,
                                    size_t argument_bytes, cl_uint work_dim,
                                    const size_t *global_offset,
                                    const size_t *global_size,
                                    const size_t *local_size);
extern cl_int __vrc_opencl_await(cl_int event);

struct _cl_platform_id {
  cl_uint active;
};
struct _cl_device_id {
  cl_uint active;
};
struct _cl_context {
  cl_uint active;
};
struct _cl_command_queue {
  cl_uint active;
  cl_context context;
};
struct _cl_program {
  cl_uint active;
  cl_context context;
  cl_build_status status;
};
struct _cl_mem {
  cl_uint active;
  cl_uint id;
  cl_context context;
  cl_mem_flags flags;
  size_t bytes;
};
struct _cl_kernel {
  cl_uint active;
  cl_uint id;
  cl_program program;
  unsigned char arguments[UDONCL_AOT_ARGUMENT_BYTES];
  cl_uint argument_offsets[UDONCL_AOT_KERNEL_CAPACITY];
  cl_uint argument_sizes[UDONCL_AOT_KERNEL_CAPACITY];
  cl_uint argument_bytes;
};

static struct _cl_platform_id udoncl_platform = {1};
static struct _cl_device_id udoncl_device = {1};
static struct _cl_context udoncl_context;
static struct _cl_command_queue udoncl_queue;
static struct _cl_program udoncl_program;
static struct _cl_mem udoncl_buffers[UDONCL_AOT_BUFFER_CAPACITY];
static struct _cl_kernel udoncl_kernels[UDONCL_AOT_KERNEL_CAPACITY];

static int udoncl_string_equal(const char *left, const char *right) {
  if (!left || !right)
    return 0;
  while (*left && *left == *right) {
    ++left;
    ++right;
  }
  return *left == *right;
}

static cl_uint udoncl_kernel_id(const char *name) {
  if (udoncl_string_equal(name, "udonblas_sgemm"))
    return 1;
  if (udoncl_string_equal(name, "udonblas_sgemv"))
    return 2;
  if (udoncl_string_equal(name, "udonblas_saxpy"))
    return 3;
  if (udoncl_string_equal(name, "udonblas_sdot_product"))
    return 4;
  if (udoncl_string_equal(name, "udonblas_sdot_reduce"))
    return 5;
  return 0;
}

static void udoncl_error(cl_int *output, cl_int value) {
  if (output)
    *output = value;
}

CL_API_ENTRY cl_int CL_API_CALL clGetPlatformIDs(cl_uint count,
                                                 cl_platform_id *platforms,
                                                 cl_uint *total) {
  if ((!platforms && !total) || (platforms && !count))
    return CL_INVALID_VALUE;
  if (platforms)
    platforms[0] = &udoncl_platform;
  if (total)
    *total = 1;
  return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL clGetDeviceIDs(cl_platform_id platform,
                                               cl_device_type type,
                                               cl_uint count,
                                               cl_device_id *devices,
                                               cl_uint *total) {
  if (platform != &udoncl_platform)
    return CL_INVALID_PLATFORM;
  if (!(type & (CL_DEVICE_TYPE_DEFAULT | CL_DEVICE_TYPE_GPU)))
    return CL_DEVICE_NOT_FOUND;
  if ((!devices && !total) || (devices && !count))
    return CL_INVALID_VALUE;
  if (devices)
    devices[0] = &udoncl_device;
  if (total)
    *total = 1;
  return CL_SUCCESS;
}

CL_API_ENTRY cl_context CL_API_CALL clCreateContext(
    const cl_context_properties *properties, cl_uint count,
    const cl_device_id *devices,
    void(CL_CALLBACK *notify)(const char *, const void *, size_t, void *),
    void *user, cl_int *error) {
  if (properties || count != 1 || !devices || devices[0] != &udoncl_device ||
      notify || user) {
    udoncl_error(error, CL_INVALID_VALUE);
    return 0;
  }
  udoncl_context.active = 1;
  udoncl_error(error, CL_SUCCESS);
  return &udoncl_context;
}

CL_API_ENTRY cl_command_queue CL_API_CALL
clCreateCommandQueue(cl_context context, cl_device_id device,
                     cl_command_queue_properties properties, cl_int *error) {
  if (context != &udoncl_context || !context->active ||
      device != &udoncl_device || properties) {
    udoncl_error(error, CL_INVALID_VALUE);
    return 0;
  }
  udoncl_queue.active = 1;
  udoncl_queue.context = context;
  udoncl_error(error, CL_SUCCESS);
  return &udoncl_queue;
}

CL_API_ENTRY cl_mem CL_API_CALL clCreateBuffer(cl_context context,
                                               cl_mem_flags flags, size_t bytes,
                                               void *host, cl_int *error) {
  cl_uint index;
  if (context != &udoncl_context || !context->active || !bytes || host) {
    udoncl_error(error, CL_INVALID_VALUE);
    return 0;
  }
  for (index = 0; index < UDONCL_AOT_BUFFER_CAPACITY; ++index) {
    cl_mem buffer = &udoncl_buffers[index];
    if (buffer->active)
      continue;
    buffer->active = 1;
    buffer->id = index + 1;
    buffer->context = context;
    buffer->flags = flags;
    buffer->bytes = bytes;
    if (__vrc_opencl_buffer_alloc(buffer->id, bytes) != CL_SUCCESS) {
      buffer->active = 0;
      udoncl_error(error, CL_OUT_OF_RESOURCES);
      return 0;
    }
    udoncl_error(error, CL_SUCCESS);
    return buffer;
  }
  udoncl_error(error, CL_OUT_OF_RESOURCES);
  return 0;
}

CL_API_ENTRY cl_program CL_API_CALL clCreateProgramWithSource(
    cl_context context, cl_uint count, const char **strings,
    const size_t *lengths, cl_int *error) {
  (void)lengths;
  if (context != &udoncl_context || !context->active || count != 1 ||
      !strings || !strings[0]) {
    udoncl_error(error, CL_INVALID_VALUE);
    return 0;
  }
  udoncl_program.active = 1;
  udoncl_program.context = context;
  udoncl_program.status = CL_BUILD_NONE;
  udoncl_error(error, CL_SUCCESS);
  return &udoncl_program;
}

CL_API_ENTRY cl_int CL_API_CALL
clBuildProgram(cl_program program, cl_uint count, const cl_device_id *devices,
               const char *options,
               void(CL_CALLBACK *notify)(cl_program, void *), void *user) {
  (void)options;
  if (program != &udoncl_program || !program->active || count || devices ||
      notify || user)
    return CL_INVALID_VALUE;
  program->status = CL_BUILD_SUCCESS;
  return CL_SUCCESS;
}

CL_API_ENTRY cl_kernel CL_API_CALL clCreateKernel(cl_program program,
                                                  const char *name,
                                                  cl_int *error) {
  cl_uint id = udoncl_kernel_id(name);
  cl_kernel kernel;
  if (program != &udoncl_program || !program->active ||
      program->status != CL_BUILD_SUCCESS || !id) {
    udoncl_error(error, id ? CL_INVALID_PROGRAM : CL_INVALID_KERNEL_NAME);
    return 0;
  }
  kernel = &udoncl_kernels[id - 1];
  kernel->active = 1;
  kernel->id = id;
  kernel->program = program;
  kernel->argument_bytes = 0;
  udoncl_error(error, CL_SUCCESS);
  return kernel;
}

CL_API_ENTRY cl_int CL_API_CALL clSetKernelArg(cl_kernel kernel, cl_uint index,
                                               size_t bytes,
                                               const void *value) {
  const unsigned char *source = (const unsigned char *)value;
  cl_uint cursor;
  if (!kernel || !kernel->active || index >= UDONCL_AOT_KERNEL_CAPACITY ||
      !bytes || !value || bytes > UDONCL_AOT_ARGUMENT_BYTES ||
      kernel->argument_bytes > UDONCL_AOT_ARGUMENT_BYTES - bytes)
    return CL_INVALID_ARG_VALUE;
  kernel->argument_offsets[index] = kernel->argument_bytes;
  kernel->argument_sizes[index] = (cl_uint)bytes;
  for (cursor = 0; cursor < bytes; ++cursor)
    kernel->arguments[kernel->argument_bytes + cursor] = source[cursor];
  kernel->argument_bytes += (cl_uint)bytes;
  return CL_SUCCESS;
}

static cl_int udoncl_wait_list(cl_uint count, const cl_event *events) {
  return (!count && !events) ? CL_SUCCESS : CL_INVALID_EVENT_WAIT_LIST;
}

CL_API_ENTRY cl_int CL_API_CALL clEnqueueWriteBuffer(
    cl_command_queue queue, cl_mem buffer, cl_bool blocking, size_t offset,
    size_t bytes, const void *source, cl_uint wait_count,
    const cl_event *wait_list, cl_event *event) {
  cl_int pending;
  if (queue != &udoncl_queue || !queue->active || !buffer || !buffer->active ||
      !source || offset > buffer->bytes || bytes > buffer->bytes - offset ||
      event || udoncl_wait_list(wait_count, wait_list) != CL_SUCCESS)
    return CL_INVALID_VALUE;
  pending = __vrc_opencl_buffer_upload(buffer->id, offset, bytes, source);
  return blocking ? __vrc_opencl_await(pending) : CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL clEnqueueReadBuffer(
    cl_command_queue queue, cl_mem buffer, cl_bool blocking, size_t offset,
    size_t bytes, void *destination, cl_uint wait_count,
    const cl_event *wait_list, cl_event *event) {
  cl_int pending;
  if (queue != &udoncl_queue || !queue->active || !buffer || !buffer->active ||
      !destination || offset > buffer->bytes ||
      bytes > buffer->bytes - offset || event ||
      udoncl_wait_list(wait_count, wait_list) != CL_SUCCESS)
    return CL_INVALID_VALUE;
  pending =
      __vrc_opencl_buffer_readback(buffer->id, offset, bytes, destination);
  return blocking ? __vrc_opencl_await(pending) : CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL clEnqueueNDRangeKernel(
    cl_command_queue queue, cl_kernel kernel, cl_uint work_dim,
    const size_t *global_offset, const size_t *global_size,
    const size_t *local_size, cl_uint wait_count, const cl_event *wait_list,
    cl_event *event) {
  if (queue != &udoncl_queue || !queue->active || !kernel || !kernel->active ||
      !global_size || !local_size || work_dim < 1 || work_dim > 3 || event ||
      udoncl_wait_list(wait_count, wait_list) != CL_SUCCESS)
    return CL_INVALID_VALUE;
  return __vrc_opencl_dispatch(kernel->id, kernel->arguments,
                               kernel->argument_bytes, work_dim, global_offset,
                               global_size, local_size) < 0
             ? CL_OUT_OF_RESOURCES
             : CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL clFinish(cl_command_queue queue) {
  return queue == &udoncl_queue && queue->active ? CL_SUCCESS
                                                 : CL_INVALID_COMMAND_QUEUE;
}

CL_API_ENTRY cl_int CL_API_CALL clReleaseKernel(cl_kernel kernel) {
  if (!kernel || !kernel->active)
    return CL_INVALID_KERNEL;
  kernel->active = 0;
  return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL clReleaseProgram(cl_program program) {
  if (program != &udoncl_program || !program->active)
    return CL_INVALID_PROGRAM;
  program->active = 0;
  return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL clReleaseMemObject(cl_mem buffer) {
  cl_int status;
  if (!buffer || !buffer->active)
    return CL_INVALID_MEM_OBJECT;
  status = __vrc_opencl_buffer_release(buffer->id);
  buffer->active = 0;
  return status;
}

CL_API_ENTRY cl_int CL_API_CALL clReleaseCommandQueue(cl_command_queue queue) {
  if (queue != &udoncl_queue || !queue->active)
    return CL_INVALID_COMMAND_QUEUE;
  queue->active = 0;
  return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL clReleaseContext(cl_context context) {
  if (context != &udoncl_context || !context->active)
    return CL_INVALID_CONTEXT;
  context->active = 0;
  return CL_SUCCESS;
}
