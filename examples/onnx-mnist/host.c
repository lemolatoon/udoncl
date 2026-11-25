#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdint.h>
#include <udoncl/export.h>
#include <udononnxruntime.h>

/* Unity imports the library under Assets/udoncl-generated/generic_onnx_mnist
 * and names its U# binding UdonclGenericOnnxMnistBinding. */
UDONCL_PROGRAM("generic_onnx_mnist");

/* The model input and output: U# writes the 28x28 image (CL_MEM_READ_ONLY)
 * with TryWriteImage before Run and reads the ten logits (CL_MEM_WRITE_ONLY)
 * with StartReadLogits after it. initialize binds them by name to the graph
 * input "image" and the graph output "logits". */
UDONCL_PUBLIC(float) cl_mem image, logits;

static const int64_t image_dims[4] = {1, 1, 28, 28};
static const int64_t logits_dims[2] = {1, 10};

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
  if (logits) {
    if (clReleaseMemObject(logits) != CL_SUCCESS)
      failed = 1;
    logits = 0;
  }
  if (image) {
    if (clReleaseMemObject(image) != CL_SUCCESS)
      failed = 1;
    image = 0;
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
  image = clCreateBuffer(context, CL_MEM_READ_ONLY, 28 * 28 * sizeof(float), 0,
                         &error);
  if (!image || error != CL_SUCCESS)
    goto fail;
  logits =
      clCreateBuffer(context, CL_MEM_WRITE_ONLY, 10 * sizeof(float), 0, &error);
  if (!logits || error != CL_SUCCESS)
    goto fail;
  status = udononnxCreateSession(context, queue, &session);
  if (status != UDONONNX_STATUS_SUCCESS)
    goto fail;
  status = udononnxBindInput(session, "image", image, 4, image_dims, 0);
  if (status != UDONONNX_STATUS_SUCCESS)
    goto fail;
  status = udononnxBindOutput(session, "logits", logits, 2, logits_dims, 0);
  if (status != UDONONNX_STATUS_SUCCESS)
    goto fail;
  return UDONONNX_STATUS_SUCCESS;
fail:
  release_resources();
  return status;
}

/* Runs the model on the bound image. It returns once the run is enqueued:
 * StartReadLogits reads through the same in-order queue. */
UDONCL_EXPORT udononnxStatus_t run(void) { return udononnxRun(session); }

/* Releases the session and the buffers; initialize may run again. */
UDONCL_EXPORT udononnxStatus_t dispose(void) {
  if (!session)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  return release_resources();
}
