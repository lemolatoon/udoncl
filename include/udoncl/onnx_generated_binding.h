#ifndef UDONCL_ONNX_GENERATED_BINDING_H
#define UDONCL_ONNX_GENERATED_BINDING_H

#include <udoncl/onnx_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Model-independent physical entry points emitted by the C++ ONNX Host
 * finalizer. User-facing C Hosts own the OpenCL context and queue and wrap
 * these with their own lifecycle API. The binding borrows both handles until
 * udon_onnx_binding_dispose returns. */
udononnxStatus_t udon_onnx_binding_create(cl_context context,
                                          cl_command_queue queue);
udononnxStatus_t udon_onnx_binding_bind_all(void);
udononnxStatus_t udon_onnx_binding_resolve_input(uint32_t input_index,
                                                 cl_mem *memory,
                                                 size_t *byte_offset,
                                                 size_t *byte_length);
udononnxStatus_t udon_onnx_binding_resolve_output(uint32_t output_index,
                                                  cl_mem *memory,
                                                  size_t *byte_offset,
                                                  size_t *byte_length);
udononnxStatus_t udon_onnx_binding_enqueue_initialize(void);
udononnxStatus_t udon_onnx_binding_enqueue_run(void);
udononnxStatus_t udon_onnx_binding_enqueue_reset_sequence_state(void);
udononnxStatus_t
udon_onnx_binding_enqueue_copy_output_to_input(uint32_t output_index,
                                               uint32_t input_index);
udononnxStatus_t udon_onnx_binding_enqueue_advance_sequence_state(void);
/* Generated only when a sequence state is banked: copies the present of
 * every banked state into its present output when the last run left it in
 * the past input buffer. Call it before reading a present output; run,
 * advance and reset never need it. */
udononnxStatus_t udon_onnx_binding_enqueue_materialize_sequence_state(void);
udononnxStatus_t udon_onnx_binding_dispose(void);

#ifdef __cplusplus
}
#endif

#endif
