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

/* Tokenizes `text` into exactly TINYLLAMA_FIXED_PREFILL_TOKEN_COUNT tokens
 * and enqueues the input_ids, attention_mask and position_ids of the prefill
 * graph into the int64 Device buffers the Host bound to those inputs. */
udononnxStatus_t
tinyllamaPrepareFixedPrefillInputs(const udontokenizers_model_t *tokenizer,
                                   udon_string_t text, cl_command_queue queue,
                                   cl_mem input_ids, cl_mem attention_mask,
                                   cl_mem position_ids);

#ifdef __cplusplus
}
#endif

#endif
