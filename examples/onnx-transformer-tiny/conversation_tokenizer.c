#include "conversation_input.h"

#include <tinyllama_tokenizer.h>

udononnxStatus_t tinyllamaPrepareGeneratedFixedPrefillInputs(
    udon_string_t text, cl_command_queue queue, cl_mem input_ids,
    cl_mem attention_mask, cl_mem position_ids) {
  udontokenizers_model_t tokenizer;
  tinyllama_tokenizerInitializeModel(&tokenizer);
  return tinyllamaPrepareFixedPrefillInputs(&tokenizer, text, queue, input_ids,
                                            attention_mask, position_ids);
}
