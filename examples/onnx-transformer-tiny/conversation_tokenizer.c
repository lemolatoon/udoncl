#include "conversation_input.h"

#include <tinyllama_tokenizer.h>

udononnxStatus_t
tinyllamaPrepareGeneratedFixedPrefillInputs(udon_string_t text,
                                            cl_command_queue queue) {
  udontokenizers_model_t tokenizer;
  tinyllama_tokenizerInitializeModel(&tokenizer);
  return tinyllamaPrepareFixedPrefillInputs(&tokenizer, text, queue);
}
