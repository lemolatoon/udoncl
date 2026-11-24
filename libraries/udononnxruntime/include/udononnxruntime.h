#ifndef UDONONNXRUNTIME_H
#define UDONONNXRUNTIME_H

/* UdonONNX: run the ONNX model a Host was built with on Device buffers the
 * Host owns. The API mirrors the ONNX Runtime C API 1.32 (OrtApi in
 * onnxruntime_c_api.h) after the Ort -> udononnx substitution:
 *
 *   udononnxCreateSession  CreateSession
 *   udononnxBindInput      CreateIoBinding + BindInput
 *   udononnxBindOutput     CreateIoBinding + BindOutput
 *   udononnxRun            RunWithBinding
 *   udononnxReleaseSession ReleaseSession
 *
 * docs/architecture/operator-ownership.md lists where it deviates: the model
 * is compiled into the Host (no environment, options or model path, and one
 * live session per Host), the session owns its binding, a tensor is a cl_mem
 * with a rank and dimensions whose element type is the model's, every call
 * returns a udononnxStatus_t, and Run returns once the graph is enqueued on
 * the session's in-order queue. */

#include <CL/cl.h>
#include <stdint.h>
#include <udoncl/onnx_types.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct udononnxSession udononnxSession_t;

/* Creates the session of the Host's model on `context` and `queue`. It
 * allocates the Device buffers the session owns and enqueues nothing; the
 * model weights are compiled into the Host. A Host has one live session:
 * a second Create before Release is UDONONNX_STATUS_INVALID_STATE. */
udononnxStatus_t udononnxCreateSession(cl_context context,
                                       cl_command_queue queue,
                                       udononnxSession_t **session);

/* Binds `memory` (owned by the caller) to the graph input or output `name`.
 * `rank` and `dims` must be the model's; `buffer_dims` is NULL or equal to
 * `dims` (another extent is UDONONNX_STATUS_NOT_SUPPORTED). Building for
 * Udon, `name` must be a compile-time constant, every graph input and output
 * must be bound, and each to exactly one buffer. */
udononnxStatus_t udononnxBindInput(udononnxSession_t *session, const char *name,
                                   cl_mem memory, int rank, const int64_t *dims,
                                   const int64_t *buffer_dims);
udononnxStatus_t udononnxBindOutput(udononnxSession_t *session,
                                    const char *name, cl_mem memory, int rank,
                                    const int64_t *dims,
                                    const int64_t *buffer_dims);

/* Runs the graph on the bound buffers: it returns once the run is enqueued
 * on the session's in-order queue, so every later command on that queue
 * observes the outputs. UDONONNX_STATUS_MISSING_BINDING while a graph input
 * or output is unbound. */
udononnxStatus_t udononnxRun(udononnxSession_t *session);

/* Releases the Device buffers the session owns and forgets the bound ones.
 * A new session may be created afterwards. */
udononnxStatus_t udononnxReleaseSession(udononnxSession_t *session);

#ifdef __cplusplus
}
#endif

#endif
