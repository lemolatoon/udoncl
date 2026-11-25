#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdint.h>
#include <udoncl/export.h>
#include <udononnxruntime.h>

/* An ONNX Host with an OpenCL C kernel of its own, shaped like the TinyLlama
 * chat example: the graph tensors live in a cl_mem array created, bound and
 * released in loops, the model input and output are bound once in
 * initialize, and the ArgMax kernel created in initialize is launched by
 * every run. */
UDONCL_PROGRAM("onnx_user_kernel");

#define TENSORS 2
#define IMAGE 0
#define LOGITS 1

static const char *const tensor_names[TENSORS] = {"image", "logits"};
static const int64_t image_dims[4] = {1, 1, 28, 28};
static const int64_t logits_dims[2] = {1, 10};
static const char kernel_source[] = "argmax.cl";

/* The bytes of each tensor, a constant of every unrolled iteration. */
#define TENSOR_BYTES(tensor)                                                   \
  (((tensor) == IMAGE ? 28 * 28 : 10) * sizeof(float))

static cl_context context;
static cl_command_queue queue;
static udononnxSession_t *session;
static cl_mem tensors[TENSORS];
static cl_mem digit;
static cl_program program;
static cl_kernel argmax;
static uint32_t predicted;
static int initialized;

UDONCL_EXPORT udononnxStatus_t initialize(void) {
  cl_platform_id platform = 0;
  cl_device_id device = 0;
  const char *source = kernel_source;
  cl_int error = CL_SUCCESS;
  udononnxStatus_t status;
  if (initialized)
    return UDONONNX_STATUS_INVALID_STATE;
  if (clGetPlatformIDs(1, &platform, 0) != CL_SUCCESS ||
      clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, 0) != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  context = clCreateContext(0, 1, &device, 0, 0, &error);
  if (error != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  queue = clCreateCommandQueue(context, device, 0, &error);
  if (error != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  for (int tensor = 0; tensor < TENSORS; ++tensor) {
    tensors[tensor] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                     TENSOR_BYTES(tensor), 0, &error);
    if (error != CL_SUCCESS)
      return UDONONNX_STATUS_OPENCL_ERROR;
  }
  digit =
      clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(uint32_t), 0, &error);
  if (error != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  status = udononnxCreateSession(context, queue, &session);
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  for (int tensor = 0; tensor < TENSORS; ++tensor) {
    if (tensor == IMAGE)
      status = udononnxBindInput(session, tensor_names[tensor], tensors[tensor],
                                 4, image_dims, 0);
    else
      status = udononnxBindOutput(session, tensor_names[tensor],
                                  tensors[tensor], 2, logits_dims, 0);
    if (status != UDONONNX_STATUS_SUCCESS)
      return status;
  }
  program = clCreateProgramWithSource(context, 1, &source, 0, &error);
  if (error != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  if (clBuildProgram(program, 0, 0, 0, 0, 0) != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  argmax = clCreateKernel(program, "mnist_argmax", &error);
  if (error != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  initialized = 1;
  return UDONONNX_STATUS_SUCCESS;
}

/* Runs the model on the image in tensors[IMAGE] and returns the predicted
 * digit, or -status. */
UDONCL_EXPORT int32_t run(void) {
  size_t one = 1;
  udononnxStatus_t status;
  if (!initialized)
    return -UDONONNX_STATUS_NOT_INITIALIZED;
  status = udononnxRun(session);
  if (status != UDONONNX_STATUS_SUCCESS)
    return -(int32_t)status;
  if (clSetKernelArg(argmax, 0, sizeof(cl_mem), &tensors[LOGITS]) !=
          CL_SUCCESS ||
      clSetKernelArg(argmax, 1, sizeof(cl_mem), &digit) != CL_SUCCESS ||
      clEnqueueNDRangeKernel(queue, argmax, 1, 0, &one, &one, 0, 0, 0) !=
          CL_SUCCESS ||
      clEnqueueReadBuffer(queue, digit, CL_TRUE, 0, sizeof(predicted),
                          &predicted, 0, 0, 0) != CL_SUCCESS)
    return -UDONONNX_STATUS_OPENCL_ERROR;
  return (int32_t)predicted;
}

UDONCL_EXPORT udononnxStatus_t dispose(void) {
  udononnxStatus_t status;
  if (!initialized)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  status = udononnxReleaseSession(session);
  clReleaseKernel(argmax);
  clReleaseProgram(program);
  clReleaseMemObject(digit);
  for (int tensor = 0; tensor < TENSORS; ++tensor)
    clReleaseMemObject(tensors[tensor]);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  initialized = 0;
  return status;
}
