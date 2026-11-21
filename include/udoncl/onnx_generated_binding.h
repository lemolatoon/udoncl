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
/* Generated only when every sequence state appends the same number of rows,
 * more than one, per run (a graph specialized to a sequence of k > 1
 * positions): advances every state by only the first `rows` (at most k) rows
 * the pending run appended, as if the run had appended those alone. The rows
 * after them, which a run computes for padded positions, are never carried:
 * the next run's past starts `rows` rows into the pending run's present.
 * rows == k is the plain advance; rows > k is UDONONNX_STATUS_INVALID_VALUE.
 * Like the plain advance it needs a pending run and leaves none. */
udononnxStatus_t
udon_onnx_binding_enqueue_advance_sequence_state_rows(uint32_t rows);
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
