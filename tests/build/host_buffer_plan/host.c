#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <udoncl/export.h>

UDONCL_PROGRAM("host_buffer_plan");

#define LAYERS 22
#define SCRATCH_BUFFERS 4

/* U# reads the value run writes. */
UDONCL_PUBLIC(float) cl_mem result;

static cl_context context;
static cl_command_queue queue;

/* One variable per buffer, reached through constant tables of their
 * addresses: udonclc unrolls the loops over the tables and folds every
 * entry to its variable. */
#define LAYER_VARIABLES(prefix)                                                \
  prefix##0, prefix##1, prefix##2, prefix##3, prefix##4, prefix##5, prefix##6, \
      prefix##7, prefix##8, prefix##9, prefix##10, prefix##11, prefix##12,     \
      prefix##13, prefix##14, prefix##15, prefix##16, prefix##17, prefix##18,  \
      prefix##19, prefix##20, prefix##21
#define LAYER_ADDRESSES(prefix)                                                \
  {                                                                            \
      &prefix##0,  &prefix##1,  &prefix##2,  &prefix##3,  &prefix##4,          \
      &prefix##5,  &prefix##6,  &prefix##7,  &prefix##8,  &prefix##9,          \
      &prefix##10, &prefix##11, &prefix##12, &prefix##13, &prefix##14,         \
      &prefix##15, &prefix##16, &prefix##17, &prefix##18, &prefix##19,         \
      &prefix##20, &prefix##21}
static cl_mem LAYER_VARIABLES(key_);
static cl_mem LAYER_VARIABLES(value_);
static cl_mem *const keys[LAYERS] = LAYER_ADDRESSES(key_);
static cl_mem *const values[LAYERS] = LAYER_ADDRESSES(value_);
static cl_mem scratch_0, scratch_1, scratch_2, scratch_3;
static cl_mem *const scratch[SCRATCH_BUFFERS] = {&scratch_0, &scratch_1,
                                                 &scratch_2, &scratch_3};

static cl_mem create(size_t bytes, cl_int *status) {
  return clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, 0, status);
}

/* SCRATCH_BUFFERS buffers, each at the clCreateBuffer of one loop. */
static void create_scratch(cl_int *status) {
  for (int index = 0; index < SCRATCH_BUFFERS; ++index)
    *scratch[index] = clCreateBuffer(context, CL_MEM_READ_WRITE, 16, 0, status);
}

UDONCL_EXPORT int initialize(void) {
  cl_platform_id platform = 0;
  cl_device_id device = 0;
  cl_int status;
  status = clGetPlatformIDs(1, &platform, 0);
  if (status != CL_SUCCESS)
    return status;
  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, 0);
  if (status != CL_SUCCESS)
    return status;
  context = clCreateContext(0, 1, &device, 0, 0, &status);
  if (status != CL_SUCCESS)
    return status;
  queue = clCreateCommandQueue(context, device, 0, &status);
  if (status != CL_SUCCESS)
    return status;
  result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 16, 0, &status);
  /* create, called twice in a loop of LAYERS iterations, allocates
   * 2 * LAYERS buffers. */
  for (int layer = 0; layer < LAYERS; ++layer) {
    *keys[layer] = create(256, &status);
    *values[layer] = create(512, &status);
  }
  create_scratch(&status);
  return status;
}

/* Fills the last layer's key and the last scratch buffer with `value` and
 * copies the scratch buffer to result. */
UDONCL_EXPORT int run(float value) {
  cl_int status = clEnqueueFillBuffer(queue, *keys[LAYERS - 1], &value,
                                      sizeof(value), 0, 256, 0, 0, 0);
  if (status == CL_SUCCESS)
    status = clEnqueueFillBuffer(queue, *scratch[SCRATCH_BUFFERS - 1], &value,
                                 sizeof(value), 0, 16, 0, 0, 0);
  if (status == CL_SUCCESS)
    status = clEnqueueCopyBuffer(queue, *scratch[SCRATCH_BUFFERS - 1], result,
                                 0, 0, 16, 0, 0, 0);
  if (status == CL_SUCCESS)
    status = clFinish(queue);
  return status;
}

UDONCL_EXPORT int dispose(void) {
  for (int index = 0; index < SCRATCH_BUFFERS; ++index)
    clReleaseMemObject(*scratch[index]);
  for (int layer = 0; layer < LAYERS; ++layer) {
    clReleaseMemObject(*keys[layer]);
    clReleaseMemObject(*values[layer]);
  }
  clReleaseMemObject(result);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  return CL_SUCCESS;
}
