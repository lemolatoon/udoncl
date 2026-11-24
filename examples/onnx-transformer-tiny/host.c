#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stddef.h>
#include <stdint.h>
#include <udoncl/export.h>
#include <udononnxruntime.h>

#include "transformer_tensors.h"

/* The Host of the synthetic transformer artifact (onnx-transformer-tiny). A
 * wrapper that includes this file for a real TinyLlama artifact
 * (prefill_host.c, step_host.c) defines its own TRANSFORMER_PROGRAM and the
 * tensor lists of its graph (transformer_tensors.h) first:
 * TRANSFORMER_INPUTS(X) and TRANSFORMER_OUTPUTS(X), and optionally
 * TRANSFORMER_EMPTY_INPUTS(X), the inputs without an element. */
#ifndef TRANSFORMER_PROGRAM
#define TRANSFORMER_PROGRAM "generic_onnx_transformer_tiny"
#define TRANSFORMER_INPUTS(X) TRANSFORMER_TINY_INPUTS(X)
#define TRANSFORMER_OUTPUTS(X) TRANSFORMER_TINY_OUTPUTS(X)
#define TRANSFORMER_COMMIT_KV 1
#endif
UDONCL_PROGRAM(TRANSFORMER_PROGRAM);

#define TRANSFORMER_BRACE(...) {__VA_ARGS__}

/* Every graph input and output is a public Device buffer: U# writes the
 * inputs (CL_MEM_READ_ONLY) before Run and reads the outputs
 * (CL_MEM_WRITE_ONLY) after it. initialize binds each to its graph tensor by
 * name. */
#define TRANSFORMER_DECLARE(variable, name, type, element_bytes, elements,     \
                            rank, dims)                                        \
  UDONCL_PUBLIC(type) cl_mem variable;                                         \
  static const int64_t variable##_dims[rank] = TRANSFORMER_BRACE dims;
TRANSFORMER_INPUTS(TRANSFORMER_DECLARE)
TRANSFORMER_OUTPUTS(TRANSFORMER_DECLARE)

#ifdef TRANSFORMER_EMPTY_INPUTS
/* A tensor without an element is bound to this private one-word buffer. */
static cl_mem transformer_empty;
#define TRANSFORMER_DECLARE_EMPTY(variable, name, type, element_bytes,         \
                                  elements, rank, dims)                        \
  static const int64_t variable##_dims[rank] = TRANSFORMER_BRACE dims;
TRANSFORMER_EMPTY_INPUTS(TRANSFORMER_DECLARE_EMPTY)
#endif

static cl_context transformer_context;
static cl_command_queue transformer_queue;
static udononnxSession_t *transformer_session;

#define TRANSFORMER_RELEASE(variable, name, type, element_bytes, elements,     \
                            rank, dims)                                        \
  if (variable) {                                                              \
    if (clReleaseMemObject(variable) != CL_SUCCESS)                            \
      failed = 1;                                                              \
    variable = 0;                                                              \
  }

static udononnxStatus_t release_resources(void) {
  int failed = 0;
  udononnxStatus_t status = UDONONNX_STATUS_SUCCESS;
  if (transformer_session) {
    status = udononnxReleaseSession(transformer_session);
    transformer_session = 0;
  }
  TRANSFORMER_OUTPUTS(TRANSFORMER_RELEASE)
  TRANSFORMER_INPUTS(TRANSFORMER_RELEASE)
#ifdef TRANSFORMER_EMPTY_INPUTS
  if (transformer_empty) {
    if (clReleaseMemObject(transformer_empty) != CL_SUCCESS)
      failed = 1;
    transformer_empty = 0;
  }
#endif
  if (transformer_queue) {
    if (clReleaseCommandQueue(transformer_queue) != CL_SUCCESS)
      failed = 1;
    transformer_queue = 0;
  }
  if (transformer_context) {
    if (clReleaseContext(transformer_context) != CL_SUCCESS)
      failed = 1;
    transformer_context = 0;
  }
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  return failed ? UDONONNX_STATUS_OPENCL_ERROR : UDONONNX_STATUS_SUCCESS;
}

#define TRANSFORMER_CREATE(flags, variable, name, type, element_bytes,         \
                           elements, rank, dims)                               \
  variable = clCreateBuffer(transformer_context, flags,                        \
                            (size_t)(elements) * (element_bytes), 0, &error);  \
  if (!variable || error != CL_SUCCESS)                                        \
    goto fail;
#define TRANSFORMER_CREATE_INPUT(...)                                          \
  TRANSFORMER_CREATE(CL_MEM_READ_ONLY, __VA_ARGS__)
#define TRANSFORMER_CREATE_OUTPUT(...)                                         \
  TRANSFORMER_CREATE(CL_MEM_WRITE_ONLY, __VA_ARGS__)
#define TRANSFORMER_BIND_INPUT(variable, name, type, element_bytes, elements,  \
                               rank, dims)                                     \
  status = udononnxBindInput(transformer_session, name, variable, rank,        \
                             variable##_dims, 0);                              \
  if (status != UDONONNX_STATUS_SUCCESS)                                       \
    goto fail;
#define TRANSFORMER_BIND_OUTPUT(variable, name, type, element_bytes, elements, \
                                rank, dims)                                    \
  status = udononnxBindOutput(transformer_session, name, variable, rank,       \
                              variable##_dims, 0);                             \
  if (status != UDONONNX_STATUS_SUCCESS)                                       \
    goto fail;
#define TRANSFORMER_BIND_EMPTY(variable, name, type, element_bytes, elements,  \
                               rank, dims)                                     \
  status = udononnxBindInput(transformer_session, name, transformer_empty,     \
                             rank, variable##_dims, 0);                        \
  if (status != UDONONNX_STATUS_SUCCESS)                                       \
    goto fail;

UDONCL_EXPORT udononnxStatus_t initialize(void) {
  cl_platform_id platform = 0;
  cl_device_id device = 0;
  cl_int error = CL_SUCCESS;
  udononnxStatus_t status = UDONONNX_STATUS_OPENCL_ERROR;
  if (transformer_session)
    return UDONONNX_STATUS_BUSY;
  if (clGetPlatformIDs(1, &platform, 0) != CL_SUCCESS ||
      clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, 0) != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  transformer_context = clCreateContext(0, 1, &device, 0, 0, &error);
  if (!transformer_context || error != CL_SUCCESS)
    goto fail;
  transformer_queue =
      clCreateCommandQueue(transformer_context, device, 0, &error);
  if (!transformer_queue || error != CL_SUCCESS)
    goto fail;
  TRANSFORMER_INPUTS(TRANSFORMER_CREATE_INPUT)
  TRANSFORMER_OUTPUTS(TRANSFORMER_CREATE_OUTPUT)
#ifdef TRANSFORMER_EMPTY_INPUTS
  transformer_empty =
      clCreateBuffer(transformer_context, CL_MEM_READ_ONLY, 4, 0, &error);
  if (!transformer_empty || error != CL_SUCCESS)
    goto fail;
#endif
  status = udononnxCreateSession(transformer_context, transformer_queue,
                                 &transformer_session);
  if (status != UDONONNX_STATUS_SUCCESS)
    goto fail;
  TRANSFORMER_INPUTS(TRANSFORMER_BIND_INPUT)
#ifdef TRANSFORMER_EMPTY_INPUTS
  TRANSFORMER_EMPTY_INPUTS(TRANSFORMER_BIND_EMPTY)
#endif
  TRANSFORMER_OUTPUTS(TRANSFORMER_BIND_OUTPUT)
  return UDONONNX_STATUS_SUCCESS;
fail:
  release_resources();
  return status;
}

/* Runs the graph once on the bound buffers. It returns once the run is
 * enqueued; every later command on the queue, a U# read included, observes
 * its outputs. */
UDONCL_EXPORT udononnxStatus_t run(void) {
  return udononnxRun(transformer_session);
}

#ifdef TRANSFORMER_COMMIT_KV
/* Carries the presents of the last run into the pasts of the next one. */
UDONCL_EXPORT udononnxStatus_t commit_kv(void) {
  if (!transformer_session)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  if (clEnqueueCopyBuffer(transformer_queue, present_key, past_key, 0, 0,
                          32 * sizeof(float), 0, 0, 0) != CL_SUCCESS ||
      clEnqueueCopyBuffer(transformer_queue, present_value, past_value, 0, 0,
                          32 * sizeof(float), 0, 0, 0) != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  return UDONONNX_STATUS_SUCCESS;
}
#endif

UDONCL_EXPORT udononnxStatus_t shutdown(void) {
  if (!transformer_session)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  return release_resources();
}
