/* The Host of the real TinyLlama M=1 step validation artifact
 * (onnx-transformer-tiny-m1-step).
 *
 * It is host.c under the program and the tensors of that graph: one position
 * on a past of TINYLLAMA_CONTEXT_CAPACITY - 1 rows. */
#include "transformer_tensors.h"

#ifndef TINYLLAMA_CONTEXT_CAPACITY
#error "TINYLLAMA_CONTEXT_CAPACITY must be defined"
#endif

#define TRANSFORMER_PROGRAM "tinyllama_real_m1_step"
#define TRANSFORMER_INPUTS(X)                                                  \
  TINYLLAMA_STEP_INPUTS(X, TINYLLAMA_CONTEXT_CAPACITY)
#define TRANSFORMER_OUTPUTS(X)                                                 \
  TINYLLAMA_STEP_OUTPUTS(X, TINYLLAMA_CONTEXT_CAPACITY)
#include "host.c"
