#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <udoncl/export.h>
#include <udoncl/onnx_generated_binding.h>

static cl_context conversation_context;
static cl_command_queue conversation_queue;
static int conversation_initialized;

static udononnxStatus_t release_opencl(void) {
  int failed = 0;
  if (conversation_queue) {
    if (clReleaseCommandQueue(conversation_queue) != CL_SUCCESS)
      failed = 1;
    conversation_queue = 0;
  }
  if (conversation_context) {
    if (clReleaseContext(conversation_context) != CL_SUCCESS)
      failed = 1;
    conversation_context = 0;
  }
  return failed ? UDONONNX_STATUS_OPENCL_ERROR : UDONONNX_STATUS_SUCCESS;
}

static udononnxStatus_t cleanup_after_failure(udononnxStatus_t status) {
  udononnxStatus_t cleanup = udon_onnx_binding_dispose();
  udononnxStatus_t opencl_cleanup = release_opencl();
  conversation_initialized = 0;
  if (cleanup != UDONONNX_STATUS_SUCCESS &&
      cleanup != UDONONNX_STATUS_NOT_INITIALIZED)
    return cleanup;
  return opencl_cleanup == UDONONNX_STATUS_SUCCESS ? status : opencl_cleanup;
}

UDONCL_EXPORT udononnxStatus_t tinyllama_conversation_initialize(void) {
  cl_platform_id platform = 0;
  cl_device_id device = 0;
  cl_int error = CL_SUCCESS;
  udononnxStatus_t status;
  if (conversation_initialized)
    return UDONONNX_STATUS_BUSY;
  if (clGetPlatformIDs(1, &platform, 0) != CL_SUCCESS ||
      clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, 0) != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  conversation_context = clCreateContext(0, 1, &device, 0, 0, &error);
  if (!conversation_context || error != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  conversation_queue =
      clCreateCommandQueue(conversation_context, device, 0, &error);
  if (!conversation_queue || error != CL_SUCCESS) {
    release_opencl();
    return UDONONNX_STATUS_OPENCL_ERROR;
  }
  status = udon_onnx_binding_create(conversation_context, conversation_queue);
  if (status != UDONONNX_STATUS_SUCCESS)
    return cleanup_after_failure(status);
  status = udon_onnx_binding_enqueue_initialize();
  if (status != UDONONNX_STATUS_SUCCESS)
    return cleanup_after_failure(status);
  status = udon_onnx_binding_bind_all();
  if (status != UDONONNX_STATUS_SUCCESS)
    return cleanup_after_failure(status);
  status = udon_onnx_binding_enqueue_reset_sequence_state();
  if (status != UDONONNX_STATUS_SUCCESS)
    return cleanup_after_failure(status);
  conversation_initialized = 1;
  return UDONONNX_STATUS_SUCCESS;
}

UDONCL_EXPORT udononnxStatus_t tinyllama_conversation_run(void) {
  if (!conversation_initialized)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  return udon_onnx_binding_enqueue_run();
}

UDONCL_EXPORT udononnxStatus_t tinyllama_conversation_dispose(void) {
  udononnxStatus_t status, opencl_status;
  if (!conversation_initialized)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  status = udon_onnx_binding_dispose();
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  opencl_status = release_opencl();
  conversation_initialized = 0;
  return opencl_status;
}
