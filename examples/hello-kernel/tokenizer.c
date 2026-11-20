#include <stdint.h>
#include <udoncl/export.h>
#include <udontokenizers.h>

static const uint32_t scalar_values[] = {0x0020, 0x65e5, 0x672c, 0x8a9e};
static const int32_t scalar_tokens[] = {13, 10, 11, 12};
static const int32_t merge_left_tokens[] = {10, 20};
static const int32_t merge_right_tokens[] = {11, 12};
static const int32_t merge_result_tokens[] = {20, 30};
static const int32_t merge_ranks[] = {0, 1};
static const int32_t byte_tokens[256] = {
    [0x8d] = 1141,
    [0x9c] = 1156,
    [0x9f] = 1159,
    [0xf0] = 1240,
};

UDONCL_EXPORT int32_t tokenizer_encode(udon_string_t text,
                                       udon_int32_array_t token_ids) {
  udontokenizers_model_t model;
  int32_t workspace[UDONTOKENIZERS_ENCODER_WORKSPACE_WORDS(64)];
  int32_t count = 0;
  udontokenizersStatus_t status;
  model.scalar_values = scalar_values;
  model.scalar_token_ids = scalar_tokens;
  model.scalar_count = 4;
  model.merge_left_token_ids = merge_left_tokens;
  model.merge_right_token_ids = merge_right_tokens;
  model.merge_result_token_ids = merge_result_tokens;
  model.merge_ranks = merge_ranks;
  model.merge_count = 2;
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
  model.merge_hash_slots = 0;
  model.merge_hash_mask = 0;
  model.merge_hash_multiplier = 0;
  status = udontokenizersEncode(
      &model, text, token_ids, udonInt32ArrayLength(token_ids), &count,
      workspace, UDONTOKENIZERS_ENCODER_WORKSPACE_WORDS(64));
  return status == UDONTOKENIZERS_STATUS_SUCCESS ? count : -(int32_t)status;
}
