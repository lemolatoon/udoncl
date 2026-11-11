#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <udoncl/export.h>
#include <udoncl/onnx_generated_binding.h>

static cl_context transformer_context;
static cl_command_queue transformer_queue;
static int transformer_initialized;

static udononnxStatus_t release_opencl(void) {
  int failed = 0;
  if (transformer_queue) {
    if (clReleaseCommandQueue(transformer_queue) != CL_SUCCESS)
      failed = 1;
    transformer_queue = 0;
  }
  if (transformer_context) {
    if (clReleaseContext(transformer_context) != CL_SUCCESS)
      failed = 1;
    transformer_context = 0;
  }
  return failed ? UDONONNX_STATUS_OPENCL_ERROR : UDONONNX_STATUS_SUCCESS;
}

static udononnxStatus_t cleanup_after_failure(udononnxStatus_t status) {
  udononnxStatus_t cleanup = udon_onnx_binding_dispose();
  udononnxStatus_t opencl_cleanup = release_opencl();
  transformer_initialized = 0;
  if (cleanup != UDONONNX_STATUS_SUCCESS &&
      cleanup != UDONONNX_STATUS_NOT_INITIALIZED)
    return cleanup;
  return opencl_cleanup == UDONONNX_STATUS_SUCCESS ? status : opencl_cleanup;
}

UDONCL_EXPORT udononnxStatus_t transformer_initialize(void) {
  cl_platform_id platform = 0;
  cl_device_id device = 0;
  cl_int error = CL_SUCCESS;
  udononnxStatus_t status;
  if (transformer_initialized)
    return UDONONNX_STATUS_BUSY;
  if (clGetPlatformIDs(1, &platform, 0) != CL_SUCCESS ||
      clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, 0) != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  transformer_context = clCreateContext(0, 1, &device, 0, 0, &error);
  if (!transformer_context || error != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  transformer_queue =
      clCreateCommandQueue(transformer_context, device, 0, &error);
  if (!transformer_queue || error != CL_SUCCESS) {
    release_opencl();
    return UDONONNX_STATUS_OPENCL_ERROR;
  }
  status = udon_onnx_binding_create(transformer_context, transformer_queue);
  if (status != UDONONNX_STATUS_SUCCESS)
    return cleanup_after_failure(status);
  status = udon_onnx_binding_enqueue_initialize();
  if (status != UDONONNX_STATUS_SUCCESS)
    return cleanup_after_failure(status);
  status = udon_onnx_binding_bind_all();
  if (status != UDONONNX_STATUS_SUCCESS)
    return cleanup_after_failure(status);
  transformer_initialized = 1;
  return UDONONNX_STATUS_SUCCESS;
}

UDONCL_EXPORT udononnxStatus_t transformer_run(void) {
  if (!transformer_initialized)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  return udon_onnx_binding_enqueue_run();
}

UDONCL_EXPORT udononnxStatus_t transformer_commit_kv(void) {
  udononnxStatus_t status;
  if (!transformer_initialized)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  status = udon_onnx_binding_enqueue_copy_output_to_input(1, 1);
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  return udon_onnx_binding_enqueue_copy_output_to_input(2, 2);
}

UDONCL_EXPORT udononnxStatus_t transformer_dispose(void) {
  udononnxStatus_t status, opencl_status;
  if (!transformer_initialized)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  status = udon_onnx_binding_dispose();
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  opencl_status = release_opencl();
  transformer_initialized = 0;
  return opencl_status;
}
