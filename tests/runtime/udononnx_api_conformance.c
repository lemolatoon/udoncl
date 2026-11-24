/* Pins the public UdonONNX API (udononnxruntime.h) to the ONNX Runtime C API
 * it mirrors: ONNX Runtime 1.32 (ORT_API_VERSION 32), OrtApi in
 * include/onnxruntime/core/session/onnxruntime_c_api.h. Each typedef below is
 * written from the ORT declaration in its comment after the Ort -> udononnx
 * substitution and the documented deviations
 * (docs/architecture/operator-ownership.md, "Device-resident model I/O"):
 *
 * - a status value instead of OrtStatus*;
 * - the model is compiled into the Host, so CreateSession takes the OpenCL
 *   context and in-order queue instead of an environment, a model path and
 *   session options;
 * - the session owns its binding (no OrtIoBinding object), and a bound
 *   tensor is a cl_mem with a rank and dimensions instead of an OrtValue
 *   (CreateTensorWithDataAsOrtValue's shape and shape_len); the element type
 *   is the model's;
 * - ReleaseSession returns a status.
 *
 * This translation unit includes nothing but the public header, so it also
 * checks that the header stands alone. */
#include <udononnxruntime.h>

/* OrtStatus* CreateSession(const OrtEnv* env, const char* model_path,
 *                          const OrtSessionOptions* options,
 *                          OrtSession** out); */
typedef udononnxStatus_t (*udononnxCreateSessionFn)(cl_context context,
                                                    cl_command_queue queue,
                                                    udononnxSession_t **out);

/* OrtStatus* BindInput(OrtIoBinding* binding_ptr, const char* name,
 *                      const OrtValue* val_ptr);
 * OrtStatus* CreateTensorWithDataAsOrtValue(const OrtMemoryInfo* info,
 *     void* p_data, size_t p_data_len, const int64_t* shape,
 *     size_t shape_len, ONNXTensorElementDataType type, OrtValue** out); */
typedef udononnxStatus_t (*udononnxBindInputFn)(udononnxSession_t *session,
                                                const char *name, cl_mem memory,
                                                int rank, const int64_t *dims,
                                                const int64_t *buffer_dims);

/* OrtStatus* BindOutput(OrtIoBinding* binding_ptr, const char* name,
 *                       const OrtValue* val_ptr); */
typedef udononnxStatus_t (*udononnxBindOutputFn)(udononnxSession_t *session,
                                                 const char *name,
                                                 cl_mem memory, int rank,
                                                 const int64_t *dims,
                                                 const int64_t *buffer_dims);

/* OrtStatus* RunWithBinding(OrtSession* session,
 *                           const OrtRunOptions* run_options,
 *                           const OrtIoBinding* binding_ptr); */
typedef udononnxStatus_t (*udononnxRunFn)(udononnxSession_t *session);

/* void ReleaseSession(OrtSession* input); */
typedef udononnxStatus_t (*udononnxReleaseSessionFn)(
    udononnxSession_t *session);

/* The status values are part of the ABI the generated Host code returns. */
_Static_assert(UDONONNX_STATUS_SUCCESS == 0, "status value");
_Static_assert(UDONONNX_STATUS_INVALID_VALUE == 1, "status value");
_Static_assert(UDONONNX_STATUS_NOT_SUPPORTED == 4, "status value");
_Static_assert(UDONONNX_STATUS_OPENCL_ERROR == 5, "status value");
_Static_assert(UDONONNX_STATUS_NOT_INITIALIZED == 6, "status value");
_Static_assert(UDONONNX_STATUS_CONTEXT_MISMATCH == 10, "status value");
_Static_assert(UDONONNX_STATUS_MISSING_BINDING == 11, "status value");
_Static_assert(UDONONNX_STATUS_INVALID_STATE == 12, "status value");

int udononnx_api_conformance(void) {
  const udononnxCreateSessionFn create = udononnxCreateSession;
  const udononnxBindInputFn bind_input = udononnxBindInput;
  const udononnxBindOutputFn bind_output = udononnxBindOutput;
  const udononnxRunFn run = udononnxRun;
  const udononnxReleaseSessionFn release = udononnxReleaseSession;
  if (!create)
    return 1;
  if (!bind_input)
    return 2;
  if (!bind_output)
    return 3;
  if (!run)
    return 4;
  if (!release)
    return 5;
  return 0;
}
