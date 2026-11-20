#include <stdint.h>
#include <tinyllama_tokenizer.h>
#include <udoncl/export.h>
#include <udontokenizers.h>

/* Pre-merge symbols one text may have; see
 * UDONTOKENIZERS_ENCODER_WORKSPACE_WORDS. */
#define TOKENIZER_SYMBOL_CAPACITY 1024

static int32_t workspace[UDONTOKENIZERS_ENCODER_WORKSPACE_WORDS(
    TOKENIZER_SYMBOL_CAPACITY)];

UDONCL_EXPORT int32_t tinyllama_tokenizer_encode(udon_string_t text,
                                                 udon_int32_array_t token_ids) {
  udontokenizers_model_t model;
  int32_t count = 0;
  tinyllama_tokenizerInitializeModel(&model);
  udontokenizersStatus_t status = udontokenizersEncode(
      &model, text, token_ids, udonInt32ArrayLength(token_ids), &count,
      workspace,
      UDONTOKENIZERS_ENCODER_WORKSPACE_WORDS(TOKENIZER_SYMBOL_CAPACITY));
  return status == UDONTOKENIZERS_STATUS_SUCCESS ? count : -(int32_t)status;
}
