#ifndef UDONCL_TINYLLAMA_CONVERSATION_INPUT_H
#define UDONCL_TINYLLAMA_CONVERSATION_INPUT_H

#include <CL/cl.h>
#include <udoncl/onnx_types.h>
#include <udontokenizers.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TINYLLAMA_FIXED_PREFILL_TOKEN_COUNT 4
/* Pre-merge symbols the fixed prefill text may have; see
 * UDONTOKENIZERS_ENCODER_WORKSPACE_WORDS. */
#define TINYLLAMA_FIXED_PREFILL_SYMBOL_CAPACITY 64

udononnxStatus_t
tinyllamaPrepareFixedPrefillInputs(const udontokenizers_model_t *tokenizer,
                                   udon_string_t text, cl_command_queue queue);

#ifdef __cplusplus
}
#endif

#endif
