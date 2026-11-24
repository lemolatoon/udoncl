#ifndef UDONONNXRUNTIME_INTERNAL_H
#define UDONONNXRUNTIME_INTERNAL_H

/* The internal surface of the UdonONNX runtime library: the model registry
 * and the generated dispatchers the linked model Host supplies, which the
 * session API (udononnxruntime.h) and the transitional binding
 * (udononnx_transitional.h) are built on. It is installed next to the
 * runtime sources, not as a public header. */

#include <stddef.h>
#include <udoncl/device_memory.h>
#include <udononnxruntime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* A graph input or output of the model. A -1 dimension is dynamic and
 * requires a positive value at the matching position in dimension_bounds.
 * The layout is UDONCL_UDON_ABI_ONNX_TENSOR_METADATA_* (the generated model
 * registry). */
typedef struct udononnxTensorMetadata {
  const char *name;
  udonclDataType_t data_type;
  uint32_t rank;
  const int64_t *dimensions;
  const int64_t *dimension_bounds;
} udononnxTensorMetadata_t;

typedef struct udononnxModel {
  const char *name;
  uint32_t graph_id;
  const udononnxTensorMetadata_t *inputs;
  size_t input_count;
  const udononnxTensorMetadata_t *outputs;
  size_t output_count;
} udononnxModel_t;

typedef struct udononnxTransitionalSession udononnxTransitionalSession_t;

/* The model registry of the linked model Host. */
size_t udononnxGeneratedGetModelCount(void);
udononnxStatus_t udononnxGeneratedGetModel(size_t index,
                                           const udononnxModel_t **model);
size_t udononnxGetModelCount(void);
udononnxStatus_t udononnxGetModel(size_t index, const udononnxModel_t **model);

/* These generated, whole-program-visible dispatchers are supplied by the
 * linked model host. They dispatch graph_id with direct calls; they are not a
 * public model-specific API or an external execution plan.
 *
 * Create allocates the Device buffers the session owns on `queue` and
 * enqueues nothing. BindMemory keeps the cl_mem a UdonONNX session binds to
 * graph input (output = 0) or output (output = 1) `index`, or forgets it for
 * a null memory; Run runs the graph on the bound buffers on `queue` and
 * enqueues a marker that reports completion. Dispose releases the session's
 * buffers and forgets every bound one. The session API passes no
 * transitional session. */
udononnxStatus_t udononnxGeneratedCreate(uint32_t graph_id,
                                         udononnxTransitionalSession_t *session,
                                         cl_command_queue queue);
udononnxStatus_t udononnxGeneratedBindMemory(uint32_t graph_id, uint32_t output,
                                             uint32_t index, cl_mem memory);
udononnxStatus_t udononnxGeneratedRun(uint32_t graph_id, cl_command_queue queue,
                                      cl_event *completion);
udononnxStatus_t
udononnxGeneratedDispose(uint32_t graph_id,
                         udononnxTransitionalSession_t *session);

/* The UdonONNX session bind of graph input or output `index`. Building for
 * Udon, udonclc folds the constant tensor name of every udononnxBindInput
 * and udononnxBindOutput call to this index. */
udononnxStatus_t udononnxBindInputAt(udononnxSession_t *session, int index,
                                     cl_mem memory, int rank,
                                     const int64_t *dims,
                                     const int64_t *buffer_dims);
udononnxStatus_t udononnxBindOutputAt(udononnxSession_t *session, int index,
                                      cl_mem memory, int rank,
                                      const int64_t *dims,
                                      const int64_t *buffer_dims);

#ifdef __cplusplus
}
#endif

#endif
