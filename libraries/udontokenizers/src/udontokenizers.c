#include <udontokenizers.h>

#define UDONTOKENIZERS_REPLACEMENT_CHARACTER 0xfffd

static int32_t scalar_token(const udontokenizers_model_t *model,
                            uint32_t scalar) {
  int32_t begin = 0;
  int32_t end = model->scalar_count;
  while (begin < end) {
    int32_t middle = begin + (end - begin) / 2;
    if (model->scalar_values[middle] < scalar)
      begin = middle + 1;
    else
      end = middle;
  }
  if (begin < model->scalar_count && model->scalar_values[begin] == scalar)
    return model->scalar_token_ids[begin];
  return -1;
}

static int32_t output_get(udon_int32_array_t managed_output,
                          int32_t *raw_output, int32_t use_raw, int32_t index) {
  return use_raw ? raw_output[index] : udonInt32ArrayGet(managed_output, index);
}

static void output_set(udon_int32_array_t managed_output, int32_t *raw_output,
                       int32_t use_raw, int32_t index, int32_t value) {
  if (use_raw)
    raw_output[index] = value;
  else
    udonInt32ArraySet(managed_output, index, value);
}

static udontokenizersStatus_t append_token(udon_int32_array_t managed_output,
                                           int32_t *raw_output, int32_t use_raw,
                                           int32_t capacity, int32_t *count,
                                           int32_t token) {
  if (*count >= capacity)
    return UDONTOKENIZERS_STATUS_OUTPUT_TOO_SMALL;
  output_set(managed_output, raw_output, use_raw, *count, token);
  ++*count;
  return UDONTOKENIZERS_STATUS_SUCCESS;
}

static udontokenizersStatus_t
append_fallback(const udontokenizers_model_t *model,
                udon_int32_array_t managed_output, int32_t *raw_output,
                int32_t use_raw, int32_t capacity, int32_t *count,
                uint32_t scalar) {
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
    udontokenizersStatus_t status =
        append_token(managed_output, raw_output, use_raw, capacity, count,
                     model->byte_token_ids[bytes[index]]);
    if (status != UDONTOKENIZERS_STATUS_SUCCESS)
      return status;
  }
  return UDONTOKENIZERS_STATUS_SUCCESS;
}

static udontokenizersStatus_t append_scalar(const udontokenizers_model_t *model,
                                            udon_int32_array_t managed_output,
                                            int32_t *raw_output,
                                            int32_t use_raw, int32_t capacity,
                                            int32_t *count, uint32_t scalar) {
  int32_t token = scalar_token(model, scalar);
  return token >= 0 ? append_token(managed_output, raw_output, use_raw,
                                   capacity, count, token)
                    : append_fallback(model, managed_output, raw_output,
                                      use_raw, capacity, count, scalar);
}

static int32_t find_merge(const udontokenizers_model_t *model, int32_t left,
                          int32_t right) {
  int32_t begin = 0;
  int32_t end = model->merge_count;
  while (begin < end) {
    int32_t middle = begin + (end - begin) / 2;
    int32_t candidate_left = model->merge_left_token_ids[middle];
    int32_t candidate_right = model->merge_right_token_ids[middle];
    if (candidate_left < left ||
        (candidate_left == left && candidate_right < right)) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  if (begin < model->merge_count &&
      model->merge_left_token_ids[begin] == left &&
      model->merge_right_token_ids[begin] == right)
    return begin;
  return -1;
}

static udontokenizersStatus_t merge_tokens(const udontokenizers_model_t *model,
                                           udon_int32_array_t managed_output,
                                           int32_t *raw_output, int32_t use_raw,
                                           int32_t first, int32_t *count) {
  for (;;) {
    int32_t best_index = -1;
    int32_t best_rank = 0x7fffffff;
    int32_t best_token = -1;
    int32_t index;
    for (index = first; index + 1 < *count; ++index) {
      int32_t left = output_get(managed_output, raw_output, use_raw, index);
      int32_t right =
          output_get(managed_output, raw_output, use_raw, index + 1);
      int32_t candidate = find_merge(model, left, right);
      if (candidate >= 0 && model->merge_ranks[candidate] < best_rank) {
        best_index = index;
        best_rank = model->merge_ranks[candidate];
        best_token = model->merge_result_token_ids[candidate];
      }
    }
    if (best_index < 0)
      return UDONTOKENIZERS_STATUS_SUCCESS;
    output_set(managed_output, raw_output, use_raw, best_index, best_token);
    for (index = best_index + 1; index + 1 < *count; ++index)
      output_set(managed_output, raw_output, use_raw, index,
                 output_get(managed_output, raw_output, use_raw, index + 1));
    --*count;
  }
}

static int32_t valid_model(const udontokenizers_model_t *model) {
  if (model->scalar_count < 0 || model->merge_count < 0 ||
      !model->scalar_values || !model->scalar_token_ids ||
      (model->merge_count &&
       (!model->merge_left_token_ids || !model->merge_right_token_ids ||
        !model->merge_result_token_ids || !model->merge_ranks)) ||
      !model->byte_token_ids || model->normalizer_prepend_scalar < -1 ||
      model->normalizer_replace_from_scalar < -1 ||
      model->normalizer_replace_to_scalar < -1 ||
      model->normalizer_prepend_scalar > 0x10ffff ||
      model->normalizer_replace_from_scalar > 0x10ffff ||
      model->normalizer_replace_to_scalar > 0x10ffff ||
      (model->normalizer_replace_from_scalar >= 0 &&
       model->normalizer_replace_to_scalar < 0) ||
      model->bos_token_id < -1 || model->added_token_count < 0 ||
      (model->added_token_count &&
       (!model->added_token_ids || !model->added_token_offsets ||
        !model->added_token_code_units)))
    return 0;
  return 1;
}

/* Text sources are either an immutable Udon string or model-owned UTF-16
 * code units. The string reference stays a function argument; it is never
 * stored in C memory. */
static int32_t source_length(udon_string_t string, const uint16_t *units,
                             int32_t units_length, int32_t use_units) {
  return use_units ? units_length : udonStringLength(string);
}

static uint32_t source_unit(udon_string_t string, const uint16_t *units,
                            int32_t use_units, int32_t index) {
  return use_units ? (uint32_t)units[index]
                   : (uint32_t)udonStringCodeUnitAt(string, index);
}

/* Returns the index of the longest added token starting at `position`, or -1.
 */
static int32_t match_added_token(const udontokenizers_model_t *model,
                                 udon_string_t string, const uint16_t *units,
                                 int32_t use_units, int32_t length,
                                 int32_t position) {
  int32_t token;
  for (token = 0; token < model->added_token_count; ++token) {
    int32_t begin = model->added_token_offsets[token];
    int32_t token_length = model->added_token_offsets[token + 1] - begin;
    int32_t index;
    if (token_length <= 0 || token_length > length - position)
      continue;
    for (index = 0; index < token_length; ++index)
      if (source_unit(string, units, use_units, position + index) !=
          (uint32_t)model->added_token_code_units[begin + index])
        break;
    if (index == token_length)
      return token;
  }
  return -1;
}

static udontokenizersStatus_t finish_segment(
    const udontokenizers_model_t *model, udontokenizers_encoder_t *encoder,
    udon_int32_array_t managed_output, int32_t *raw_output, int32_t use_raw) {
  udontokenizersStatus_t status = UDONTOKENIZERS_STATUS_SUCCESS;
  if (encoder->segment_started)
    status = merge_tokens(model, managed_output, raw_output, use_raw,
                          encoder->segment_first, &encoder->count);
  encoder->segment_first = encoder->count;
  encoder->segment_started = 0;
  return status;
}

static udontokenizersStatus_t
encoder_append(const udontokenizers_model_t *model,
               udontokenizers_encoder_t *encoder, udon_string_t string,
               const uint16_t *units, int32_t units_length, int32_t use_units,
               udon_int32_array_t managed_output, int32_t *raw_output,
               int32_t use_raw) {
  int32_t length;
  int32_t position;
  udontokenizersStatus_t status;
  if (!model || !encoder)
    return UDONTOKENIZERS_STATUS_INVALID_ARGUMENT;
  if (encoder->status != UDONTOKENIZERS_STATUS_SUCCESS)
    return (udontokenizersStatus_t)encoder->status;
#if !(defined(__clang__) && defined(__SPIR__))
  if ((use_units ? (!units && units_length) : !string) ||
      (use_raw ? !raw_output : !managed_output))
    return UDONTOKENIZERS_STATUS_INVALID_ARGUMENT;
#endif
  if (use_units && units_length < 0)
    return UDONTOKENIZERS_STATUS_INVALID_ARGUMENT;
  length = source_length(string, units, units_length, use_units);
  for (position = 0; position < length; ++position) {
    uint32_t scalar;
    int32_t added =
        match_added_token(model, string, units, use_units, length, position);
    if (added >= 0) {
      status =
          finish_segment(model, encoder, managed_output, raw_output, use_raw);
      if (status == UDONTOKENIZERS_STATUS_SUCCESS)
        status =
            append_token(managed_output, raw_output, use_raw, encoder->capacity,
                         &encoder->count, model->added_token_ids[added]);
      if (status != UDONTOKENIZERS_STATUS_SUCCESS) {
        encoder->status = status;
        return status;
      }
      encoder->segment_first = encoder->count;
      position += model->added_token_offsets[added + 1] -
                  model->added_token_offsets[added] - 1;
      continue;
    }
    scalar = source_unit(string, units, use_units, position);
    if (scalar >= 0xd800 && scalar <= 0xdbff) {
      uint32_t low;
      if (position + 1 >= length) {
        encoder->status = UDONTOKENIZERS_STATUS_INVALID_UTF16;
        return UDONTOKENIZERS_STATUS_INVALID_UTF16;
      }
      low = source_unit(string, units, use_units, ++position);
      if (low < 0xdc00 || low > 0xdfff) {
        encoder->status = UDONTOKENIZERS_STATUS_INVALID_UTF16;
        return UDONTOKENIZERS_STATUS_INVALID_UTF16;
      }
      scalar = 0x10000 + ((scalar - 0xd800) << 10) + (low - 0xdc00);
    } else if (scalar >= 0xdc00 && scalar <= 0xdfff) {
      encoder->status = UDONTOKENIZERS_STATUS_INVALID_UTF16;
      return UDONTOKENIZERS_STATUS_INVALID_UTF16;
    }
    if (!encoder->segment_started) {
      encoder->segment_started = 1;
      encoder->segment_first = encoder->count;
      if (model->normalizer_prepend_scalar >= 0) {
        status = append_scalar(model, managed_output, raw_output, use_raw,
                               encoder->capacity, &encoder->count,
                               (uint32_t)model->normalizer_prepend_scalar);
        if (status != UDONTOKENIZERS_STATUS_SUCCESS) {
          encoder->status = status;
          return status;
        }
      }
    }
    if (model->normalizer_replace_from_scalar >= 0 &&
        scalar == (uint32_t)model->normalizer_replace_from_scalar)
      scalar = (uint32_t)model->normalizer_replace_to_scalar;
    status = append_scalar(model, managed_output, raw_output, use_raw,
                           encoder->capacity, &encoder->count, scalar);
    if (status != UDONTOKENIZERS_STATUS_SUCCESS) {
      encoder->status = status;
      return status;
    }
  }
  return UDONTOKENIZERS_STATUS_SUCCESS;
}

static udontokenizersStatus_t encoder_begin(
    const udontokenizers_model_t *model, udontokenizers_encoder_t *encoder,
    udon_int32_array_t managed_output, int32_t *raw_output, int32_t use_raw,
    int32_t output_capacity, int32_t add_special_tokens) {
  if (!model || !encoder || output_capacity < 0)
    return UDONTOKENIZERS_STATUS_INVALID_ARGUMENT;
  encoder->count = 0;
  encoder->capacity = output_capacity;
  encoder->segment_first = 0;
  encoder->segment_started = 0;
  encoder->status = UDONTOKENIZERS_STATUS_SUCCESS;
  if (!valid_model(model)) {
    encoder->status = UDONTOKENIZERS_STATUS_INVALID_MODEL;
    return UDONTOKENIZERS_STATUS_INVALID_MODEL;
  }
  if (add_special_tokens && model->bos_token_id >= 0) {
    udontokenizersStatus_t status =
        append_token(managed_output, raw_output, use_raw, output_capacity,
                     &encoder->count, model->bos_token_id);
    if (status != UDONTOKENIZERS_STATUS_SUCCESS) {
      encoder->status = status;
      return status;
    }
  }
  encoder->segment_first = encoder->count;
  return UDONTOKENIZERS_STATUS_SUCCESS;
}

static udontokenizersStatus_t
encoder_finish(const udontokenizers_model_t *model,
               udontokenizers_encoder_t *encoder,
               udon_int32_array_t managed_output, int32_t *raw_output,
               int32_t use_raw, int32_t *output_count) {
  udontokenizersStatus_t status;
  if (!model || !encoder || !output_count)
    return UDONTOKENIZERS_STATUS_INVALID_ARGUMENT;
  if (encoder->status != UDONTOKENIZERS_STATUS_SUCCESS)
    return (udontokenizersStatus_t)encoder->status;
  status = finish_segment(model, encoder, managed_output, raw_output, use_raw);
  if (status != UDONTOKENIZERS_STATUS_SUCCESS) {
    encoder->status = status;
    return status;
  }
  *output_count = encoder->count;
  return UDONTOKENIZERS_STATUS_SUCCESS;
}

static udontokenizersStatus_t
encode(const udontokenizers_model_t *model, udon_string_t input,
       udon_int32_array_t managed_output, int32_t *raw_output, int32_t use_raw,
       int32_t output_capacity, int32_t *output_count) {
  udontokenizers_encoder_t encoder;
  udontokenizersStatus_t status;
  if (!model || !output_count || output_capacity < 0)
    return UDONTOKENIZERS_STATUS_INVALID_ARGUMENT;
#if !(defined(__clang__) && defined(__SPIR__))
  if (!input || (use_raw ? !raw_output : !managed_output))
    return UDONTOKENIZERS_STATUS_INVALID_ARGUMENT;
#endif
  status = encoder_begin(model, &encoder, managed_output, raw_output, use_raw,
                         output_capacity, 1);
  if (status == UDONTOKENIZERS_STATUS_SUCCESS)
    status = encoder_append(model, &encoder, input, 0, 0, 0, managed_output,
                            raw_output, use_raw);
  if (status == UDONTOKENIZERS_STATUS_SUCCESS)
    status = encoder_finish(model, &encoder, managed_output, raw_output,
                            use_raw, output_count);
  return status;
}

udontokenizersStatus_t udontokenizersEncode(const udontokenizers_model_t *model,
                                            udon_string_t input,
                                            udon_int32_array_t output,
                                            int32_t output_capacity,
                                            int32_t *output_count) {
  if (output_capacity < 0 || output_capacity > udonInt32ArrayLength(output))
    return UDONTOKENIZERS_STATUS_INVALID_ARGUMENT;
  return encode(model, input, output, 0, 0, output_capacity, output_count);
}

udontokenizersStatus_t
udontokenizersEncodeRaw(const udontokenizers_model_t *model,
                        udon_string_t input, int32_t *output,
                        int32_t output_capacity, int32_t *output_count) {
  return encode(model, input, (udon_int32_array_t)0, output, 1, output_capacity,
                output_count);
}

udontokenizersStatus_t udontokenizersEncoderBeginRaw(
    const udontokenizers_model_t *model, udontokenizers_encoder_t *encoder,
    int32_t *output, int32_t output_capacity, int32_t add_special_tokens) {
  if (!output)
    return UDONTOKENIZERS_STATUS_INVALID_ARGUMENT;
  return encoder_begin(model, encoder, (udon_int32_array_t)0, output, 1,
                       output_capacity, add_special_tokens);
}

udontokenizersStatus_t
udontokenizersEncoderAppendRaw(const udontokenizers_model_t *model,
                               udontokenizers_encoder_t *encoder,
                               udon_string_t text, int32_t *output) {
  return encoder_append(model, encoder, text, 0, 0, 0, (udon_int32_array_t)0,
                        output, 1);
}

udontokenizersStatus_t udontokenizersEncoderAppendCodeUnitsRaw(
    const udontokenizers_model_t *model, udontokenizers_encoder_t *encoder,
    const uint16_t *code_units, int32_t length, int32_t *output) {
  return encoder_append(model, encoder, (udon_string_t)0, code_units, length, 1,
                        (udon_int32_array_t)0, output, 1);
}

udontokenizersStatus_t
udontokenizersEncoderFinishRaw(const udontokenizers_model_t *model,
                               udontokenizers_encoder_t *encoder,
                               int32_t *output, int32_t *output_count) {
  return encoder_finish(model, encoder, (udon_int32_array_t)0, output, 1,
                        output_count);
}

static udontokenizersStatus_t
append_chat_literal(const udontokenizers_model_t *model,
                    const udontokenizers_chat_template_t *chat,
                    udontokenizers_encoder_t *encoder, int32_t offset,
                    int32_t length, int32_t *output) {
  if (!length)
    return UDONTOKENIZERS_STATUS_SUCCESS;
  return encoder_append(model, encoder, (udon_string_t)0,
                        chat->code_units + offset, length, 1,
                        (udon_int32_array_t)0, output, 1);
}

udontokenizersStatus_t udontokenizersEncoderAppendChatMessageRaw(
    const udontokenizers_model_t *model,
    const udontokenizers_chat_template_t *chat,
    udontokenizers_encoder_t *encoder, int32_t role, udon_string_t content,
    int32_t *output) {
  udontokenizersStatus_t status;
  if (!chat || !chat->code_units || !chat->role_prefix_offsets ||
      !chat->role_prefix_lengths || !chat->role_suffix_offsets ||
      !chat->role_suffix_lengths || role < 0 ||
      role >= UDONTOKENIZERS_CHAT_ROLE_COUNT)
    return UDONTOKENIZERS_STATUS_INVALID_ARGUMENT;
  if (chat->role_prefix_lengths[role] < 0 ||
      chat->role_suffix_lengths[role] < 0)
    return UDONTOKENIZERS_STATUS_INVALID_ARGUMENT;
  status =
      append_chat_literal(model, chat, encoder, chat->role_prefix_offsets[role],
                          chat->role_prefix_lengths[role], output);
  if (status == UDONTOKENIZERS_STATUS_SUCCESS)
    status = udontokenizersEncoderAppendRaw(model, encoder, content, output);
  if (status == UDONTOKENIZERS_STATUS_SUCCESS)
    status = append_chat_literal(model, chat, encoder,
                                 chat->role_suffix_offsets[role],
                                 chat->role_suffix_lengths[role], output);
  return status;
}

udontokenizersStatus_t udontokenizersEncoderAppendChatGenerationPromptRaw(
    const udontokenizers_model_t *model,
    const udontokenizers_chat_template_t *chat,
    udontokenizers_encoder_t *encoder, int32_t *output) {
  if (!chat || !chat->code_units || chat->generation_prompt_length < 0)
    return UDONTOKENIZERS_STATUS_INVALID_ARGUMENT;
  return append_chat_literal(model, chat, encoder,
                             chat->generation_prompt_offset,
                             chat->generation_prompt_length, output);
}

static udontokenizersStatus_t emit_unit(udontokenizers_decoder_t *decoder,
                                        int32_t *output, int32_t unit) {
  if (decoder->count >= decoder->capacity)
    return UDONTOKENIZERS_STATUS_OUTPUT_TOO_SMALL;
  output[decoder->count++] = unit;
  return UDONTOKENIZERS_STATUS_SUCCESS;
}

static udontokenizersStatus_t emit_scalar(udontokenizers_decoder_t *decoder,
                                          int32_t *output, int32_t scalar) {
  udontokenizersStatus_t status;
  if (scalar < 0x10000)
    return emit_unit(decoder, output, scalar);
  status = emit_unit(decoder, output, 0xd800 + ((scalar - 0x10000) >> 10));
  if (status == UDONTOKENIZERS_STATUS_SUCCESS)
    status = emit_unit(decoder, output, 0xdc00 + ((scalar - 0x10000) & 0x3ff));
  return status;
}

/* A byte-fallback run that is not valid UTF-8 decodes to one replacement
 * character per byte. Rewrite the run's output from its first unit. */
static udontokenizersStatus_t invalidate_run(udontokenizers_decoder_t *decoder,
                                             int32_t *output) {
  int32_t index;
  if (decoder->run_start + decoder->run_bytes > decoder->capacity)
    return UDONTOKENIZERS_STATUS_OUTPUT_TOO_SMALL;
  for (index = 0; index < decoder->run_bytes; ++index)
    output[decoder->run_start + index] = UDONTOKENIZERS_REPLACEMENT_CHARACTER;
  decoder->count = decoder->run_start + decoder->run_bytes;
  decoder->run_invalid = 1;
  decoder->pending_remaining = 0;
  return UDONTOKENIZERS_STATUS_SUCCESS;
}

static udontokenizersStatus_t end_run(udontokenizers_decoder_t *decoder,
                                      int32_t *output) {
  udontokenizersStatus_t status = UDONTOKENIZERS_STATUS_SUCCESS;
  if (decoder->run_bytes && !decoder->run_invalid && decoder->pending_remaining)
    status = invalidate_run(decoder, output);
  decoder->run_bytes = 0;
  decoder->run_invalid = 0;
  decoder->pending_remaining = 0;
  return status;
}

static udontokenizersStatus_t decode_byte(udontokenizers_decoder_t *decoder,
                                          int32_t *output, int32_t byte) {
  if (!decoder->run_bytes)
    decoder->run_start = decoder->count;
  ++decoder->run_bytes;
  if (decoder->run_invalid)
    return emit_unit(decoder, output, UDONTOKENIZERS_REPLACEMENT_CHARACTER);
  if (decoder->pending_remaining) {
    if (byte < decoder->pending_lower || byte > decoder->pending_upper)
      return invalidate_run(decoder, output);
    decoder->pending_scalar = (decoder->pending_scalar << 6) | (byte & 0x3f);
    decoder->pending_lower = 0x80;
    decoder->pending_upper = 0xbf;
    if (--decoder->pending_remaining)
      return UDONTOKENIZERS_STATUS_SUCCESS;
    return emit_scalar(decoder, output, decoder->pending_scalar);
  }
  decoder->pending_lower = 0x80;
  decoder->pending_upper = 0xbf;
  if (byte <= 0x7f)
    return emit_unit(decoder, output, byte);
  if (byte >= 0xc2 && byte <= 0xdf) {
    decoder->pending_scalar = byte & 0x1f;
    decoder->pending_remaining = 1;
  } else if (byte >= 0xe0 && byte <= 0xef) {
    decoder->pending_scalar = byte & 0x0f;
    decoder->pending_remaining = 2;
    if (byte == 0xe0)
      decoder->pending_lower = 0xa0;
    else if (byte == 0xed)
      decoder->pending_upper = 0x9f;
  } else if (byte >= 0xf0 && byte <= 0xf4) {
    decoder->pending_scalar = byte & 0x07;
    decoder->pending_remaining = 3;
    if (byte == 0xf0)
      decoder->pending_lower = 0x90;
    else if (byte == 0xf4)
      decoder->pending_upper = 0x8f;
  } else {
    return invalidate_run(decoder, output);
  }
  return UDONTOKENIZERS_STATUS_SUCCESS;
}

udontokenizersStatus_t udontokenizersDecoderBeginRaw(
    const udontokenizers_model_t *model, udontokenizers_decoder_t *decoder,
    int32_t *output, int32_t output_capacity, int32_t skip_special_tokens) {
  if (!model || !decoder || !output || output_capacity < 0)
    return UDONTOKENIZERS_STATUS_INVALID_ARGUMENT;
  decoder->count = 0;
  decoder->capacity = output_capacity;
  decoder->skip_special_tokens = skip_special_tokens ? 1 : 0;
  decoder->run_start = 0;
  decoder->run_bytes = 0;
  decoder->run_invalid = 0;
  decoder->pending_scalar = 0;
  decoder->pending_remaining = 0;
  decoder->pending_lower = 0x80;
  decoder->pending_upper = 0xbf;
  decoder->status = UDONTOKENIZERS_STATUS_SUCCESS;
  if (model->vocab_count <= 0 || !model->piece_offsets ||
      !model->piece_code_units || !model->piece_kinds ||
      model->decoder_strip_scalar < -1 ||
      model->decoder_strip_scalar > 0xffff || model->decoder_strip_count < 0) {
    decoder->status = UDONTOKENIZERS_STATUS_INVALID_MODEL;
    return UDONTOKENIZERS_STATUS_INVALID_MODEL;
  }
  return UDONTOKENIZERS_STATUS_SUCCESS;
}

udontokenizersStatus_t
udontokenizersDecoderAppendRaw(const udontokenizers_model_t *model,
                               udontokenizers_decoder_t *decoder,
                               int32_t token_id, int32_t *output) {
  int32_t kind;
  int32_t index;
  udontokenizersStatus_t status;
  if (!model || !decoder || !output)
    return UDONTOKENIZERS_STATUS_INVALID_ARGUMENT;
  if (decoder->status != UDONTOKENIZERS_STATUS_SUCCESS)
    return (udontokenizersStatus_t)decoder->status;
  if (token_id < 0 || token_id >= model->vocab_count) {
    decoder->status = UDONTOKENIZERS_STATUS_INVALID_TOKEN;
    return UDONTOKENIZERS_STATUS_INVALID_TOKEN;
  }
  kind = model->piece_kinds[token_id];
  if (kind == UDONTOKENIZERS_PIECE_SPECIAL && decoder->skip_special_tokens)
    return UDONTOKENIZERS_STATUS_SUCCESS;
  if (kind >= 0) {
    status = decode_byte(decoder, output, kind & 0xff);
  } else {
    status = end_run(decoder, output);
    for (index = model->piece_offsets[token_id];
         status == UDONTOKENIZERS_STATUS_SUCCESS &&
         index < model->piece_offsets[token_id + 1];
         ++index)
      status = emit_unit(decoder, output, model->piece_code_units[index]);
  }
  if (status != UDONTOKENIZERS_STATUS_SUCCESS)
    decoder->status = status;
  return status;
}

udontokenizersStatus_t
udontokenizersDecoderFinishRaw(const udontokenizers_model_t *model,
                               udontokenizers_decoder_t *decoder,
                               int32_t *output, int32_t *output_count) {
  int32_t strip = 0;
  int32_t index;
  udontokenizersStatus_t status;
  if (!model || !decoder || !output || !output_count)
    return UDONTOKENIZERS_STATUS_INVALID_ARGUMENT;
  if (decoder->status != UDONTOKENIZERS_STATUS_SUCCESS)
    return (udontokenizersStatus_t)decoder->status;
  status = end_run(decoder, output);
  if (status != UDONTOKENIZERS_STATUS_SUCCESS) {
    decoder->status = status;
    return status;
  }
  if (model->decoder_strip_scalar >= 0)
    while (strip < model->decoder_strip_count && strip < decoder->count &&
           output[strip] == model->decoder_strip_scalar)
      ++strip;
  for (index = strip; index < decoder->count; ++index)
    output[index - strip] = output[index];
  decoder->count -= strip;
  *output_count = decoder->count;
  return UDONTOKENIZERS_STATUS_SUCCESS;
}
