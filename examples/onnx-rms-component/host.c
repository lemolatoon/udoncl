#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <udoncl/export.h>
#include <udoncl/onnx_generated_binding.h>

/* Unity imports the component under
 * Assets/udoncl-generated/onnx_rms_component. */
UDONCL_PROGRAM("onnx_rms_component");

static cl_context context;
static cl_command_queue queue;
static int initialized;

static udononnxStatus_t release_opencl(void) {
  int failed = 0;
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
  return failed ? UDONONNX_STATUS_OPENCL_ERROR : UDONONNX_STATUS_SUCCESS;
}

static udononnxStatus_t cleanup_after_failure(udononnxStatus_t status) {
  udononnxStatus_t binding_status = udon_onnx_binding_dispose();
  udononnxStatus_t opencl_status = release_opencl();
  initialized = 0;
  if (binding_status != UDONONNX_STATUS_SUCCESS &&
      binding_status != UDONONNX_STATUS_NOT_INITIALIZED)
    return binding_status;
  return opencl_status == UDONONNX_STATUS_SUCCESS ? status : opencl_status;
}

UDONCL_EXPORT udononnxStatus_t initialize(void) {
  cl_platform_id platform = 0;
  cl_device_id device = 0;
  cl_int error = CL_SUCCESS;
  udononnxStatus_t status;
  if (initialized)
    return UDONONNX_STATUS_BUSY;
  if (clGetPlatformIDs(1, &platform, 0) != CL_SUCCESS ||
      clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, 0) != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  context = clCreateContext(0, 1, &device, 0, 0, &error);
  if (!context || error != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  queue = clCreateCommandQueue(context, device, 0, &error);
  if (!queue || error != CL_SUCCESS) {
    release_opencl();
    return UDONONNX_STATUS_OPENCL_ERROR;
  }
  status = udon_onnx_binding_create(context, queue);
  if (status != UDONONNX_STATUS_SUCCESS)
    return cleanup_after_failure(status);
  status = udon_onnx_binding_enqueue_initialize();
  if (status != UDONONNX_STATUS_SUCCESS)
    return cleanup_after_failure(status);
  status = udon_onnx_binding_bind_all();
  if (status != UDONONNX_STATUS_SUCCESS)
    return cleanup_after_failure(status);
  initialized = 1;
  return UDONONNX_STATUS_SUCCESS;
}

UDONCL_EXPORT udononnxStatus_t run(void) {
  if (!initialized)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  return udon_onnx_binding_enqueue_run();
}

UDONCL_EXPORT udononnxStatus_t shutdown(void) {
  udononnxStatus_t status;
  udononnxStatus_t opencl_status;
  if (!initialized)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  status = udon_onnx_binding_dispose();
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  opencl_status = release_opencl();
  initialized = 0;
  return opencl_status;
}
