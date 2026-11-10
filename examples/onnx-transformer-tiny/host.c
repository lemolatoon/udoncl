#include <udoncl/export.h>
#include <udoncl/onnx_generated_binding.h>

static int transformer_initialized;

static udononnxStatus_t cleanup_after_failure(udononnxStatus_t status) {
  udononnxStatus_t cleanup = udon_onnx_binding_dispose();
  transformer_initialized = 0;
  return cleanup == UDONONNX_STATUS_SUCCESS ||
                 cleanup == UDONONNX_STATUS_NOT_INITIALIZED
             ? status
             : cleanup;
}

UDONCL_EXPORT udononnxStatus_t transformer_initialize(void) {
  udononnxStatus_t status;
  if (transformer_initialized)
    return UDONONNX_STATUS_BUSY;
  status = udon_onnx_binding_create();
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

UDONCL_EXPORT udononnxStatus_t transformer_dispose(void) {
  udononnxStatus_t status;
  if (!transformer_initialized)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  status = udon_onnx_binding_dispose();
  if (status == UDONONNX_STATUS_SUCCESS)
    transformer_initialized = 0;
  return status;
}
