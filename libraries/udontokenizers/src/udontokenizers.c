#include <udontokenizers.h>

static int32_t scalar_token(const udontokenizers_model_t *model,
                            uint32_t scalar) {
  int32_t index;
  for (index = 0; index < model->scalar_count; ++index)
    if (model->scalar_values[index] == scalar)
      return model->scalar_token_ids[index];
  return -1;
}

static udontokenizersStatus_t append_token(udon_int32_array_t output,
                                           int32_t capacity, int32_t *count,
                                           int32_t token) {
  if (*count >= capacity)
    return UDONTOKENIZERS_STATUS_OUTPUT_TOO_SMALL;
  udonInt32ArraySet(output, *count, token);
  ++*count;
  return UDONTOKENIZERS_STATUS_SUCCESS;
}

static udontokenizersStatus_t
append_fallback(const udontokenizers_model_t *model, udon_int32_array_t output,
                int32_t capacity, int32_t *count, uint32_t scalar) {
  uint8_t bytes[4];
  int32_t length;
  int32_t index;
  if (scalar <= 0x7f) {
    bytes[0] = (uint8_t)scalar;
    length = 1;
  } else if (scalar <= 0x7ff) {
    bytes[0] = (uint8_t)(0xc0 | (scalar >> 6));
    bytes[1] = (uint8_t)(0x80 | (scalar & 0x3f));
    length = 2;
  } else if (scalar <= 0xffff) {
    bytes[0] = (uint8_t)(0xe0 | (scalar >> 12));
    bytes[1] = (uint8_t)(0x80 | ((scalar >> 6) & 0x3f));
    bytes[2] = (uint8_t)(0x80 | (scalar & 0x3f));
    length = 3;
  } else {
    bytes[0] = (uint8_t)(0xf0 | (scalar >> 18));
    bytes[1] = (uint8_t)(0x80 | ((scalar >> 12) & 0x3f));
    bytes[2] = (uint8_t)(0x80 | ((scalar >> 6) & 0x3f));
    bytes[3] = (uint8_t)(0x80 | (scalar & 0x3f));
    length = 4;
  }
  for (index = 0; index < length; ++index) {
    udontokenizersStatus_t status = append_token(
        output, capacity, count, model->byte_token_ids[bytes[index]]);
    if (status != UDONTOKENIZERS_STATUS_SUCCESS)
      return status;
  }
  return UDONTOKENIZERS_STATUS_SUCCESS;
}

static udontokenizersStatus_t merge_tokens(const udontokenizers_model_t *model,
                                           udon_int32_array_t output,
                                           int32_t *count) {
  for (;;) {
    int32_t best_index = -1;
    int32_t best_rank = 0x7fffffff;
    int32_t best_token = -1;
    int32_t index;
    int32_t merge;
    for (index = 0; index + 1 < *count; ++index) {
      int32_t left = udonInt32ArrayGet(output, index);
      int32_t right = udonInt32ArrayGet(output, index + 1);
      for (merge = 0; merge < model->merge_count; ++merge) {
        const udontokenizers_merge_t *candidate = &model->merges[merge];
        if (candidate->left_token_id == left &&
            candidate->right_token_id == right && candidate->rank < best_rank) {
          best_index = index;
          best_rank = candidate->rank;
          best_token = candidate->result_token_id;
        }
      }
    }
    if (best_index < 0)
      return UDONTOKENIZERS_STATUS_SUCCESS;
    udonInt32ArraySet(output, best_index, best_token);
    for (index = best_index + 1; index + 1 < *count; ++index)
      udonInt32ArraySet(output, index, udonInt32ArrayGet(output, index + 1));
    --*count;
  }
}

udontokenizersStatus_t udontokenizersEncode(const udontokenizers_model_t *model,
                                            udon_string_t input,
                                            udon_int32_array_t output,
                                            int32_t output_capacity,
                                            int32_t *output_count) {
  int32_t input_length;
  int32_t position;
  int32_t count = 0;
  if (!model || !output_count || output_capacity < 0 ||
      output_capacity > udonInt32ArrayLength(output))
    return UDONTOKENIZERS_STATUS_INVALID_ARGUMENT;
#if !(defined(__clang__) && defined(__SPIR__))
  if (!input || !output)
    return UDONTOKENIZERS_STATUS_INVALID_ARGUMENT;
#endif
  if (model->scalar_count < 0 || model->merge_count < 0 ||
      !model->scalar_values || !model->scalar_token_ids ||
      (model->merge_count && !model->merges) || !model->byte_token_ids)
    return UDONTOKENIZERS_STATUS_INVALID_MODEL;

  input_length = udonStringLength(input);
  for (position = 0; position < input_length; ++position) {
    uint32_t scalar = (uint32_t)udonStringCodeUnitAt(input, position);
    int32_t token;
    udontokenizersStatus_t status;
    if (scalar >= 0xd800 && scalar <= 0xdbff) {
      uint32_t low;
      if (position + 1 >= input_length)
        return UDONTOKENIZERS_STATUS_INVALID_UTF16;
      low = (uint32_t)udonStringCodeUnitAt(input, ++position);
      if (low < 0xdc00 || low > 0xdfff)
        return UDONTOKENIZERS_STATUS_INVALID_UTF16;
      scalar = 0x10000 + ((scalar - 0xd800) << 10) + (low - 0xdc00);
    } else if (scalar >= 0xdc00 && scalar <= 0xdfff) {
      return UDONTOKENIZERS_STATUS_INVALID_UTF16;
    }
    token = scalar_token(model, scalar);
    status = token >= 0 ? append_token(output, output_capacity, &count, token)
                        : append_fallback(model, output, output_capacity,
                                          &count, scalar);
    if (status != UDONTOKENIZERS_STATUS_SUCCESS)
      return status;
  }
  *output_count = count;
  return merge_tokens(model, output, output_count);
}
