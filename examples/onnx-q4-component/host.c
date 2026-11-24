#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdint.h>
#include <udoncl/export.h>
#include <udononnxruntime.h>

/* Unity imports the component under
 * Assets/udoncl-generated/onnx_q4_component. */
UDONCL_PROGRAM("onnx_q4_component");

/* The graph input and output ([4, 2048] -> [4, 33]): U# writes `input`
 * (CL_MEM_READ_ONLY) before Run and reads `output` (CL_MEM_WRITE_ONLY)
 * after it. initialize binds them by name. */
UDONCL_PUBLIC(float) cl_mem input, output;

static const int64_t input_dims[2] = {4, 2048};
static const int64_t output_dims[2] = {4, 33};

static cl_context context;
static cl_command_queue queue;
static udononnxSession_t *session;

static udononnxStatus_t release_resources(void) {
  int failed = 0;
  udononnxStatus_t status = UDONONNX_STATUS_SUCCESS;
  if (session) {
    status = udononnxReleaseSession(session);
    session = 0;
  }
  if (output) {
    if (clReleaseMemObject(output) != CL_SUCCESS)
      failed = 1;
    output = 0;
  }
  if (input) {
    if (clReleaseMemObject(input) != CL_SUCCESS)
      failed = 1;
    input = 0;
  }
  if (queue) {
    if (clReleaseCommandQueue(queue) != CL_SUCCESS)
      failed = 1;
    queue = 0;
  }
  if (context) {
    if (clReleaseContext(context) != CL_SUCCESS)
      failed = 1;
    context = 0;
  }
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  return failed ? UDONONNX_STATUS_OPENCL_ERROR : UDONONNX_STATUS_SUCCESS;
}

UDONCL_EXPORT udononnxStatus_t initialize(void) {
  cl_platform_id platform = 0;
  cl_device_id device = 0;
  cl_int error = CL_SUCCESS;
  udononnxStatus_t status = UDONONNX_STATUS_OPENCL_ERROR;
  if (session)
    return UDONONNX_STATUS_BUSY;
  if (clGetPlatformIDs(1, &platform, 0) != CL_SUCCESS ||
      clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, 0) != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  context = clCreateContext(0, 1, &device, 0, 0, &error);
  if (!context || error != CL_SUCCESS)
    goto fail;
  queue = clCreateCommandQueue(context, device, 0, &error);
  if (!queue || error != CL_SUCCESS)
    goto fail;
  input = clCreateBuffer(context, CL_MEM_READ_ONLY, 4 * 2048 * sizeof(float), 0,
                         &error);
  if (!input || error != CL_SUCCESS)
    goto fail;
  output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 4 * 33 * sizeof(float), 0,
                          &error);
  if (!output || error != CL_SUCCESS)
    goto fail;
  status = udononnxCreateSession(context, queue, &session);
  if (status != UDONONNX_STATUS_SUCCESS)
    goto fail;
  status = udononnxBindInput(session, "input", input, 2, input_dims, 0);
  if (status != UDONONNX_STATUS_SUCCESS)
    goto fail;
  status = udononnxBindOutput(session, "output", output, 2, output_dims, 0);
  if (status != UDONONNX_STATUS_SUCCESS)
    goto fail;
  return UDONONNX_STATUS_SUCCESS;
fail:
  release_resources();
  return status;
}

UDONCL_EXPORT udononnxStatus_t run(void) { return udononnxRun(session); }

UDONCL_EXPORT udononnxStatus_t shutdown(void) {
  if (!session)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  return release_resources();
}
