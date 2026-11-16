#ifndef UDONCL_ONNX_BINDING_ABI_H
#define UDONCL_ONNX_BINDING_ABI_H

#include <stddef.h>
#include <stdint.h>
#include <udoncl/onnx_types.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct udononnxSession udononnxSession_t;
typedef struct udononnxModel udononnxModel_t;

#define UDONONNX_BINDING_METADATA_STORAGE_BYTES 64u
#define UDONONNX_BINDING_MODEL_STORAGE_BYTES 64u
#define UDONONNX_BINDING_STATE_STORAGE_BYTES 256u
#define UDONONNX_BINDING_TENSOR_STORAGE_BYTES 128u

/* Internal physical ABI used by generated U# bindings. This is deliberately
 * flattened to values supported by the Udon heap boundary. C callers should
 * construct udonclBufferView_t and udononnxTensor_t directly instead. */
udononnxStatus_t udononnxBindingCreateTensor(
    uint32_t buffer_slot, uint32_t buffer_generation, uint32_t byte_offset,
    uint32_t byte_length, udonclDataType_t data_type, uint32_t rank,
    const int32_t *dimensions, udononnxTensor_t *tensor);

udononnxStatus_t udononnxBindingInitializeTensorMetadata(
    void *metadata_storage, uint32_t metadata_storage_bytes, uint32_t index,
    const char *name, udonclDataType_t data_type, uint32_t rank,
    const int64_t *dimensions, const int64_t *dimension_bounds);

udononnxStatus_t udononnxBindingInitializeModel(
    void *model_storage, uint32_t model_storage_bytes, const char *name,
    uint32_t graph_id, void *input_metadata, uint32_t input_count,
    void *output_metadata, uint32_t output_count,
    const udononnxModel_t **model);

udononnxStatus_t udononnxBindingBindInput(
    udononnxSession_t *session, uint32_t input_index, uint32_t buffer_slot,
    uint32_t buffer_generation, uint32_t byte_offset, uint32_t byte_length,
    udonclDataType_t data_type, uint32_t rank, const int32_t *dimensions);

udononnxStatus_t udononnxBindingBindOutput(
    udononnxSession_t *session, uint32_t output_index, uint32_t buffer_slot,
    uint32_t buffer_generation, uint32_t byte_offset, uint32_t byte_length,
    udonclDataType_t data_type, uint32_t rank, const int32_t *dimensions);

udononnxStatus_t
udononnxBindingStateCreate(void *state_storage, uint32_t state_storage_bytes,
                           cl_context context, cl_command_queue queue,
                           uint32_t model_index, void *input_bindings,
                           uint32_t input_binding_bytes, void *output_bindings,
                           uint32_t output_binding_bytes);
udononnxStatus_t udononnxBindingStateBindInput(
    void *state_storage, uint32_t input_index, uint32_t buffer_slot,
    uint32_t buffer_generation, uint32_t byte_offset, uint32_t byte_length,
    udonclDataType_t data_type, uint32_t rank, const int32_t *dimensions);
udononnxStatus_t udononnxBindingStateBindOutput(
    void *state_storage, uint32_t output_index, uint32_t buffer_slot,
    uint32_t buffer_generation, uint32_t byte_offset, uint32_t byte_length,
    udonclDataType_t data_type, uint32_t rank, const int32_t *dimensions);
udononnxStatus_t
udononnxBindingStateAllocateBuffer(void *state_storage, uint32_t access,
                                   uint32_t byte_capacity,
                                   udonclBufferHandle_t *buffer);
udononnxStatus_t
udononnxBindingStateBindTensor(void *state_storage, uint32_t output,
                               uint32_t tensor_index, uint32_t buffer_slot,
                               uint32_t buffer_generation, uint32_t byte_offset,
                               uint32_t byte_length, udonclDataType_t data_type,
                               uint32_t rank, const int32_t *dimensions);
udononnxStatus_t
udononnxBindingStateEnqueueInitialize(void *state_storage, uint32_t wait_count,
                                      const cl_event *wait_list,
                                      cl_event *completion);
udononnxStatus_t udononnxBindingStateEnqueueRun(void *state_storage,
                                                uint32_t wait_count,
                                                const cl_event *wait_list,
                                                cl_event *completion);
udononnxStatus_t udononnxBindingStateGetSession(void *state_storage,
                                                udononnxSession_t **session);
/* Generated AOT bind-all fast path. Begin validates the same session lifecycle
 * accepted by per-tensor bind calls; Complete records the final status. */
udononnxStatus_t udononnxBindingStateBeginStaticBind(void *state_storage);
udononnxStatus_t
udononnxBindingStateCompleteStaticBind(void *state_storage,
                                       udononnxStatus_t status);
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
udononnxStatus_t udononnxBindingStateCompleteEvent(void *state_storage,
                                                   cl_event completion,
                                                   udononnxStatus_t result);
udononnxStatus_t udononnxBindingStateGetContext(void *state_storage,
                                                cl_context *context);
udononnxStatus_t udononnxBindingStateGetCommandQueue(void *state_storage,
                                                     cl_command_queue *queue);
udononnxStatus_t udononnxBindingStateDispose(void *state_storage);

#ifdef __cplusplus
}
#endif

#endif
