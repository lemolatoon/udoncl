#ifndef UDONONNX_TRANSITIONAL_H
#define UDONONNX_TRANSITIONAL_H

/* TRANSITIONAL, NOT PUBLIC API: removed in stage 3.
 *
 * The runtime side of the private sequence-state binding the ONNX Host
 * compiler generates for a model library built with
 * TRANSITIONAL_SEQUENCE_STATE (the __udononnx_transitional_* functions the
 * TinyLlama Conversation and Chat Hosts call). The generated binding keeps
 * its state in Host globals (__udon_onnx_binding_state, the binding tables
 * __udon_onnx_binding_inputs/outputs) and calls the udononnxBindingState*
 * functions below; its run dispatcher reads the bound tensors through
 * udononnxSessionResolveInput/Output and caches what it resolved under
 * udononnxSessionBindingVersion. Everything else is private to
 * udononnx_transitional.c. */

#include "udononnxruntime_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

/* A tensor of the binding table: a non-owning view of a Device buffer with
 * the tensor's rank and dimensions (UDONCL_UDON_ABI_ONNX_TENSOR_*, which the
 * generated bind-all writes directly). */
typedef struct udononnxTensor {
  udonclBufferView_t storage;
  uint32_t rank;
  int64_t dimensions[UDONONNX_MAX_TENSOR_RANK];
} udononnxTensor_t;

typedef enum udononnxTransitionalSessionState {
  UDONONNX_SESSION_EMPTY = 0,
  UDONONNX_SESSION_CREATED = 1,
  UDONONNX_SESSION_INITIALIZING = 2,
  UDONONNX_SESSION_READY = 3,
  UDONONNX_SESSION_RUNNING = 4,
  UDONONNX_SESSION_DISPOSED = 5
} udononnxTransitionalSessionState_t;

/* The session of the binding: the model, the binding tables (borrowed from
 * the generated binding) and the run state. The first run after the tables
 * change validates the whole table; later runs of the unchanged table only
 * check that every bound buffer is still alive, which is known without a
 * check while no Device buffer handle was destroyed since
 * (udonclDeviceBufferEpoch).
 *
 * binding_version identifies the binding table: creating the session and
 * every invalidation give it a value no session had before, so the
 * generated run, which resolves the bound buffers to physical tokens once,
 * can reuse them while the version is equal. */
struct udononnxTransitionalSession {
  const udononnxModel_t *model;
  cl_context context;
  udononnxTensor_t *input_bindings;
  udononnxTensor_t *output_bindings;
  uint32_t generation;
  uint32_t operation_generation;
  udononnxStatus_t last_status;
  cl_event active_event;
  udononnxTransitionalSessionState_t state;
  uint32_t bindings_validated;
  uint32_t validated_buffer_epoch;
  uint32_t binding_version;
};

/* Bytes of the Host globals the generated binding keeps (ONNXHostABI.h). */
#define UDONONNX_BINDING_STATE_STORAGE_BYTES 256u
#define UDONONNX_BINDING_TENSOR_STORAGE_BYTES 128u

udononnxStatus_t
udononnxBindingStateCreate(void *state_storage, uint32_t state_storage_bytes,
                           cl_context context, cl_command_queue queue,
                           uint32_t model_index, void *input_bindings,
                           uint32_t input_binding_bytes, void *output_bindings,
                           uint32_t output_binding_bytes);
udononnxStatus_t
udononnxBindingStateAllocateBuffer(void *state_storage, uint32_t access,
                                   uint32_t byte_capacity,
                                   udonclBufferHandle_t *buffer);
/* The generated bind-all writes the binding tables itself between Begin,
 * which checks the session can be bound, and Complete, which records its
 * status. */
udononnxStatus_t udononnxBindingStateBeginStaticBind(void *state_storage);
udononnxStatus_t
udononnxBindingStateCompleteStaticBind(void *state_storage,
                                       udononnxStatus_t status);
udononnxStatus_t
udononnxBindingStateEnqueueInitialize(void *state_storage, uint32_t wait_count,
                                      const cl_event *wait_list,
                                      cl_event *completion);
udononnxStatus_t udononnxBindingStateEnqueueRun(void *state_storage,
                                                uint32_t wait_count,
                                                const cl_event *wait_list,
                                                cl_event *completion);
udononnxStatus_t
udononnxBindingStateGetSession(void *state_storage,
                               udononnxTransitionalSession_t **session);
udononnxStatus_t udononnxBindingStateResolveInput(void *state_storage,
                                                  uint32_t input_index,
                                                  cl_mem *memory,
                                                  size_t *byte_offset,
                                                  size_t *byte_length);
udononnxStatus_t udononnxBindingStateResolveOutput(void *state_storage,
                                                   uint32_t output_index,
                                                   cl_mem *memory,
                                                   size_t *byte_offset,
                                                   size_t *byte_length);
/* Called by the generated continuation that observes the completion of an
 * enqueued initialize or run. It never releases the event. */
udononnxStatus_t udononnxBindingStateCompleteEvent(void *state_storage,
                                                   cl_event completion,
                                                   udononnxStatus_t result);
udononnxStatus_t udononnxBindingStateGetCommandQueue(void *state_storage,
                                                     cl_command_queue *queue);
udononnxStatus_t udononnxBindingStateDispose(void *state_storage);

/* The bound tensors, for the generated run. */
udononnxStatus_t
udononnxSessionResolveInput(const udononnxTransitionalSession_t *session,
                            size_t index, cl_mem *memory, size_t *byte_offset,
                            size_t *byte_length);
udononnxStatus_t
udononnxSessionResolveOutput(const udononnxTransitionalSession_t *session,
                             size_t index, cl_mem *memory, size_t *byte_offset,
                             size_t *byte_length);
/* The current binding version (see udononnxTransitionalSession_t); 0
 * without a session. */
uint32_t
udononnxSessionBindingVersion(const udononnxTransitionalSession_t *session);

/* The generated dispatchers of the transitional session. EnqueueInitialize
 * is Create followed by a marker that reports completion; EnqueueRun runs
 * the graph on the bound tables and enqueues a marker. */
udononnxStatus_t udononnxGeneratedEnqueueInitialize(
    uint32_t graph_id, udononnxTransitionalSession_t *session,
    cl_command_queue queue, cl_uint wait_count, const cl_event *wait_list,
    cl_event *completion);
udononnxStatus_t
udononnxGeneratedEnqueueRun(uint32_t graph_id,
                            udononnxTransitionalSession_t *session,
                            cl_command_queue queue, cl_uint wait_count,
                            const cl_event *wait_list, cl_event *completion);

#ifdef __cplusplus
}
#endif

#endif
