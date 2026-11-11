#include <stdint.h>
#include <udoncl/export.h>
#include <udontokenizers.h>

static const uint32_t scalar_values[] = {0x0020, 0x65e5, 0x672c, 0x8a9e};
static const int32_t scalar_tokens[] = {13, 10, 11, 12};
static const udontokenizers_merge_t merges[] = {
    {10, 11, 20, 0},
    {20, 12, 30, 1},
};
static const int32_t byte_tokens[256] = {
    [0x8d] = 1141,
    [0x9c] = 1156,
    [0x9f] = 1159,
    [0xf0] = 1240,
};

UDONCL_EXPORT int32_t tokenizer_encode(udon_string_t text,
                                       udon_int32_array_t token_ids) {
  udontokenizers_model_t model;
  int32_t count = 0;
  udontokenizersStatus_t status;
  model.scalar_values = scalar_values;
  model.scalar_token_ids = scalar_tokens;
  model.scalar_count = 4;
  model.merges = merges;
  model.merge_count = 2;
  model.byte_token_ids = byte_tokens;
  status = udontokenizersEncode(&model, text, token_ids,
                                udonInt32ArrayLength(token_ids), &count);
  return status == UDONTOKENIZERS_STATUS_SUCCESS ? count : -(int32_t)status;
}
