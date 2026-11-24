/* The Host of the real TinyLlama prefill validation artifact
 * (onnx-transformer-tiny built with UDONCL_TINYLLAMA_ONNX_MODEL).
 *
 * It is host.c under the program and the tensors of that graph: four
 * positions on an empty past, whose 44 past K/V inputs hold no element. */
#include "transformer_tensors.h"

#define TRANSFORMER_PROGRAM "tinyllama_real_prefill"
#define TRANSFORMER_INPUTS(X) TINYLLAMA_PREFILL_INPUTS(X)
#define TRANSFORMER_EMPTY_INPUTS(X) TINYLLAMA_PREFILL_EMPTY_INPUTS(X)
#define TRANSFORMER_OUTPUTS(X) TINYLLAMA_PREFILL_OUTPUTS(X)
#include "host.c"
