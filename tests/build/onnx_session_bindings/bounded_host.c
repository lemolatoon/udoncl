#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdint.h>
#include <udoncl/export.h>
#include <udononnxruntime.h>

UDONCL_PROGRAM("onnx_session_bindings_bounded");

/* present = Concat(past, step) along the bounded dimension `past` (at most
 * 15 rows of 8 floats): U# writes past and step and reads present. Every
 * run binds the run-time extents again; past and present are the leading
 * blocks of buffers of CAPACITY rows. */
UDONCL_PUBLIC(float) cl_mem past, step;
UDONCL_PUBLIC(float) cl_mem present;

#define CAPACITY 16
#define WIDTH 8

static const int64_t step_dims[2] = {1, WIDTH};
static const int64_t buffer_dims[2] = {CAPACITY, WIDTH};

static cl_context context;
static cl_command_queue queue;
static udononnxSession_t *session;

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
  past = clCreateBuffer(context, CL_MEM_READ_ONLY,
                        CAPACITY * WIDTH * sizeof(float), 0, &error);
  if (error != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  step = clCreateBuffer(context, CL_MEM_READ_ONLY, WIDTH * sizeof(float), 0,
                        &error);
  if (error != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  present = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                           CAPACITY * WIDTH * sizeof(float), 0, &error);
  if (error != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  status = udononnxCreateSession(context, queue, &session);
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  return udononnxBindInput(session, "step", step, 2, step_dims, 0);
}

/* Runs the graph on a past of `length` rows. */
UDONCL_EXPORT udononnxStatus_t run(int length) {
  int64_t past_dims[2] = {length, WIDTH};
  int64_t present_dims[2] = {length + 1, WIDTH};
  udononnxStatus_t status =
      udononnxBindInput(session, "past", past, 2, past_dims, buffer_dims);
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  status = udononnxBindOutput(session, "present", present, 2, present_dims,
                              buffer_dims);
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  return udononnxRun(session);
}

UDONCL_EXPORT udononnxStatus_t shutdown(void) {
  udononnxStatus_t status = udononnxReleaseSession(session);
  session = 0;
  clReleaseMemObject(present);
  clReleaseMemObject(step);
  clReleaseMemObject(past);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  return status;
}
