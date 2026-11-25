#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdint.h>
#include <udoncl/export.h>
#include <udononnxruntime.h>

UDONCL_PROGRAM("onnx_session_bindings");

/* sum = ((a + b) + c)^T of 32 x 256 floats: U# writes a, b and c and reads
 * sum. */
UDONCL_PUBLIC(float) cl_mem a, b, c;
UDONCL_PUBLIC(float) cl_mem sum;

#define ELEMENTS (32 * 256)

static const int64_t dims[2] = {32, 256};
static const int64_t sum_dims[2] = {256, 32};
/* The inputs are bound in a loop over constant tables of their names and
 * variables; udonclc unrolls it and folds every name to its graph input. */
static const char *const input_names[3] = {"a", "b", "c"};
static cl_mem *const input_buffers[3] = {&a, &b, &c};

static cl_context context;
static cl_command_queue queue;
static udononnxSession_t *session;

static cl_mem create(cl_mem_flags flags, cl_int *error) {
  return clCreateBuffer(context, flags, ELEMENTS * sizeof(float), 0, error);
}

UDONCL_EXPORT udononnxStatus_t initialize(void) {
  cl_platform_id platform = 0;
  cl_device_id device = 0;
  cl_int error = CL_SUCCESS;
  udononnxStatus_t status;
  if (session)
    return UDONONNX_STATUS_BUSY;
  if (clGetPlatformIDs(1, &platform, 0) != CL_SUCCESS ||
      clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, 0) != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  context = clCreateContext(0, 1, &device, 0, 0, &error);
  if (!context || error != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  queue = clCreateCommandQueue(context, device, 0, &error);
  if (!queue || error != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  a = create(CL_MEM_READ_ONLY, &error);
  if (error != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  b = create(CL_MEM_READ_ONLY, &error);
  if (error != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  c = create(CL_MEM_READ_ONLY, &error);
  if (error != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  sum = create(CL_MEM_WRITE_ONLY, &error);
  if (error != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  status = udononnxCreateSession(context, queue, &session);
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  for (int i = 0; i < 3; ++i) {
    status = udononnxBindInput(session, input_names[i], *input_buffers[i], 2,
                               dims, 0);
    if (status != UDONONNX_STATUS_SUCCESS)
      return status;
  }
  return udononnxBindOutput(session, "sum", sum, 2, sum_dims, sum_dims);
}

UDONCL_EXPORT udononnxStatus_t run(void) { return udononnxRun(session); }

UDONCL_EXPORT udononnxStatus_t dispose(void) {
  udononnxStatus_t status = udononnxReleaseSession(session);
  session = 0;
  clReleaseMemObject(sum);
  clReleaseMemObject(c);
  clReleaseMemObject(b);
  clReleaseMemObject(a);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  return status;
}
