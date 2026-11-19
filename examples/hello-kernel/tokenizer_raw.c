#include <stdint.h>
#include <udoncl/export.h>
#include <udontokenizers.h>

static const uint32_t scalar_values[] = {0x0061};
static const int32_t scalar_tokens[] = {7};
static const int32_t byte_tokens[256] = {0};

UDONCL_EXPORT int32_t tokenizer_raw_first(udon_string_t text) {
  udontokenizers_model_t model;
  int32_t token_ids[4];
  int32_t count = 0;
  udontokenizersStatus_t status;
  model.scalar_values = scalar_values;
  model.scalar_token_ids = scalar_tokens;
  model.scalar_count = 1;
  model.merge_left_token_ids = 0;
  model.merge_right_token_ids = 0;
  model.merge_result_token_ids = 0;
  model.merge_ranks = 0;
  model.merge_count = 0;
  model.byte_token_ids = byte_tokens;
  model.normalizer_prepend_scalar = -1;
  model.normalizer_replace_from_scalar = -1;
  model.normalizer_replace_to_scalar = -1;
  model.bos_token_id = -1;
  model.added_token_ids = 0;
  model.added_token_offsets = 0;
  model.added_token_code_units = 0;
  model.added_token_count = 0;
  model.piece_offsets = 0;
  model.piece_code_units = 0;
  model.piece_kinds = 0;
  model.vocab_count = 0;
  model.decoder_strip_scalar = -1;
  model.decoder_strip_count = 0;
  model.eos_token_id = -1;
  status = udontokenizersEncodeRaw(&model, text, token_ids, 4, &count);
  if (status != UDONTOKENIZERS_STATUS_SUCCESS)
    return -(int32_t)status;
  return count > 0 ? token_ids[0] : -1;
}
