#ifndef UDONCL_ONNX_GENERATED_BINDING_H
#define UDONCL_ONNX_GENERATED_BINDING_H

#include <udoncl/onnx_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Model-independent physical entry points emitted by the C++ ONNX Host
 * finalizer. User-facing C Hosts wrap these with their own lifecycle API. */
udononnxStatus_t udon_onnx_binding_create(void);
udononnxStatus_t udon_onnx_binding_bind_all(void);
udononnxStatus_t udon_onnx_binding_enqueue_initialize(void);
udononnxStatus_t udon_onnx_binding_enqueue_run(void);
udononnxStatus_t
udon_onnx_binding_enqueue_copy_output_to_input(uint32_t output_index,
                                               uint32_t input_index);
udononnxStatus_t udon_onnx_binding_dispose(void);

#ifdef __cplusplus
}
#endif

#endif
