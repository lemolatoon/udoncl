#ifndef UDONCL_TINYLLAMA_TRANSITIONAL_SEQUENCE_STATE_H
#define UDONCL_TINYLLAMA_TRANSITIONAL_SEQUENCE_STATE_H

/* TRANSITIONAL, NOT PUBLIC API: removed in stage 3.
 *
 * The private binding the ONNX Host compiler generates for a model library
 * built with TRANSITIONAL_SEQUENCE_STATE (udoncl_add_onnx_host_library),
 * which only the TinyLlama Conversation and Chat model libraries of this
 * example are. It keeps the KV cache inside the binding (banked sequence
 * state) until these Hosts bind it as ordinary graph inputs and outputs
 * through the public UdonONNX API (udononnxruntime.h) with in-place Concat
 * and bounded dynamic dimensions. Nothing outside conversation_host.c may
 * use it.
 *
 * The binding borrows the OpenCL context and queue the Host passes to create
 * until dispose returns. */

#include <CL/cl.h>
#include <stddef.h>
#include <stdint.h>
#include <udoncl/onnx_types.h>

#ifdef __cplusplus
extern "C" {
#endif

udononnxStatus_t __udononnx_transitional_create(cl_context context,
                                                cl_command_queue queue);
udononnxStatus_t __udononnx_transitional_enqueue_initialize(void);
udononnxStatus_t __udononnx_transitional_bind_all(void);
udononnxStatus_t __udononnx_transitional_resolve_input(uint32_t input_index,
                                                       cl_mem *memory,
                                                       size_t *byte_offset,
                                                       size_t *byte_length);
udononnxStatus_t __udononnx_transitional_resolve_output(uint32_t output_index,
                                                        cl_mem *memory,
                                                        size_t *byte_offset,
                                                        size_t *byte_length);
udononnxStatus_t __udononnx_transitional_enqueue_run(void);
udononnxStatus_t __udononnx_transitional_reset_sequence_state(void);
udononnxStatus_t __udononnx_transitional_advance_sequence_state(void);
/* Generated only when every sequence state appends the same number of rows,
 * more than one, per run (a graph specialized to a sequence of k > 1
 * positions): advances every state by only the first `rows` (at most k) rows
 * the pending run appended, as if the run had appended those alone. The rows
 * after them, which a run computes for padded positions, are never carried:
 * the next run's past starts `rows` rows into the pending run's present.
 * rows == k is the plain advance; rows > k is UDONONNX_STATUS_INVALID_VALUE.
 * Like the plain advance it needs a pending run and leaves none. */
udononnxStatus_t
__udononnx_transitional_advance_sequence_state_rows(uint32_t rows);
udononnxStatus_t __udononnx_transitional_dispose(void);

#ifdef __cplusplus
}
#endif

#endif
