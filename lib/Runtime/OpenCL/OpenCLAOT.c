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
#define UDONCL_AOT_ARGUMENT_BYTES 8

/* The compiler backend returns an opaque physical-buffer token.  It is kept
 * separately from the OpenCL handle-table id: cl_mem identity and lifetime
 * belong to this runtime, while the token only selects a preallocated Unity
 * backing resource.  Zero is reserved for allocation failure. */
extern cl_uint __vrc_opencl_buffer_alloc(cl_uint logical_id, size_t bytes);
extern cl_int __vrc_opencl_buffer_release(cl_uint physical_token);
extern cl_int __vrc_opencl_buffer_upload(cl_uint physical_token, size_t offset,
                                         size_t bytes, const void *source);
extern cl_int __vrc_opencl_buffer_readback(cl_uint physical_token,
                                           size_t offset, size_t bytes,
                                           void *destination);
extern cl_int __vrc_opencl_dispatch(cl_uint kernel_id,
                                    const unsigned char *arguments,
                                    size_t argument_bytes, cl_uint work_dim,
                                    const size_t *global_offset,
                                    const size_t *global_size,
                                    const size_t *local_size);
extern cl_int __vrc_opencl_dispatch_udonblas_sgemm(
    cl_uint input_token, cl_uint output_token, cl_int m, cl_int n, cl_int k,
    cl_int transa, cl_int transb, cl_int lda, cl_int ldb, cl_int ldc,
    cl_float alpha, cl_float beta);
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
  cl_uint physical_token;
  cl_context context;
  cl_mem_flags flags;
  size_t bytes;
};
struct _cl_kernel {
  cl_uint active;
  cl_uint id;
  cl_program program;
  unsigned char arguments[UDONCL_AOT_KERNEL_CAPACITY]
                         [UDONCL_AOT_ARGUMENT_BYTES];
  cl_uint argument_sizes[UDONCL_AOT_KERNEL_CAPACITY];
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
    buffer->physical_token =
        __vrc_opencl_buffer_alloc(buffer->id, buffer->bytes);
    if (!buffer->physical_token) {
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
  for (id = 0; id < UDONCL_AOT_KERNEL_CAPACITY; ++id)
    kernel->argument_sizes[id] = 0;
  udoncl_error(error, CL_SUCCESS);
  return kernel;
}

CL_API_ENTRY cl_int CL_API_CALL clSetKernelArg(cl_kernel kernel, cl_uint index,
                                               size_t bytes,
                                               const void *value) {
  const unsigned char *source = (const unsigned char *)value;
  cl_uint cursor;
  if (!kernel || !kernel->active || index >= UDONCL_AOT_KERNEL_CAPACITY ||
      !bytes || !value || bytes > UDONCL_AOT_ARGUMENT_BYTES)
    return CL_INVALID_ARG_VALUE;
  kernel->argument_sizes[index] = (cl_uint)bytes;
  for (cursor = 0; cursor < bytes; ++cursor)
    kernel->arguments[index][cursor] = source[cursor];
  return CL_SUCCESS;
}

static cl_int udoncl_i32_argument(cl_kernel kernel, cl_uint index,
                                  cl_int *value) {
  unsigned char *destination = (unsigned char *)value;
  cl_uint cursor;
  if (kernel->argument_sizes[index] != sizeof(*value))
    return CL_INVALID_KERNEL_ARGS;
  for (cursor = 0; cursor < sizeof(*value); ++cursor)
    destination[cursor] = kernel->arguments[index][cursor];
  return CL_SUCCESS;
}

static cl_int udoncl_f32_argument(cl_kernel kernel, cl_uint index,
                                  cl_float *value) {
  unsigned char *destination = (unsigned char *)value;
  cl_uint cursor;
  if (kernel->argument_sizes[index] != sizeof(*value))
    return CL_INVALID_KERNEL_ARGS;
  for (cursor = 0; cursor < sizeof(*value); ++cursor)
    destination[cursor] = kernel->arguments[index][cursor];
  return CL_SUCCESS;
}

static cl_int udoncl_mem_argument(cl_kernel kernel, cl_uint index,
                                  cl_mem *value) {
  unsigned char *destination = (unsigned char *)value;
  cl_uint cursor;
  if (kernel->argument_sizes[index] != sizeof(*value))
    return CL_INVALID_KERNEL_ARGS;
  for (cursor = 0; cursor < sizeof(*value); ++cursor)
    destination[cursor] = kernel->arguments[index][cursor];
  if (!*value || !(*value)->active)
    return CL_INVALID_MEM_OBJECT;
  return CL_SUCCESS;
}

static cl_int udoncl_dispatch_sgemm(cl_kernel kernel, cl_uint work_dim,
                                    const size_t *global_offset,
                                    const size_t *global_size,
                                    const size_t *local_size) {
  cl_mem input, output;
  cl_int values[8];
  cl_float alpha, beta;
  cl_uint index;
  if (work_dim != 2 || global_offset || global_size[0] != 32 ||
      global_size[1] != 32 || local_size[0] != 1 || local_size[1] != 1)
    return CL_INVALID_GLOBAL_WORK_SIZE;
  if (udoncl_mem_argument(kernel, 0, &input) != CL_SUCCESS ||
      udoncl_mem_argument(kernel, 1, &output) != CL_SUCCESS)
    return CL_INVALID_MEM_OBJECT;
  for (index = 0; index < 8; ++index)
    if (udoncl_i32_argument(kernel, index + 2, &values[index]) != CL_SUCCESS)
      return CL_INVALID_KERNEL_ARGS;
  if (udoncl_f32_argument(kernel, 10, &alpha) != CL_SUCCESS ||
      udoncl_f32_argument(kernel, 11, &beta) != CL_SUCCESS)
    return CL_INVALID_KERNEL_ARGS;
  return __vrc_opencl_dispatch_udonblas_sgemm(
      input->physical_token, output->physical_token, values[0], values[1],
      values[2], values[3], values[4], values[5], values[6], values[7], alpha,
      beta);
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
  pending =
      __vrc_opencl_buffer_upload(buffer->physical_token, offset, bytes, source);
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
  pending = __vrc_opencl_buffer_readback(buffer->physical_token, offset, bytes,
                                         destination);
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
  if (kernel->id == 1)
    return udoncl_dispatch_sgemm(kernel, work_dim, global_offset, global_size,
                                 local_size) < 0
               ? CL_OUT_OF_RESOURCES
               : CL_SUCCESS;
  return __vrc_opencl_dispatch(kernel->id, &kernel->arguments[0][0],
                               sizeof(kernel->arguments), work_dim,
                               global_offset, global_size, local_size) < 0
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
  status = __vrc_opencl_buffer_release(buffer->physical_token);
  buffer->physical_token = 0;
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
