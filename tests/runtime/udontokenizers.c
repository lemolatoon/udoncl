#include <stdint.h>
#include <stdio.h>
#include <udoncl/array.h>
#include <udoncl/string.h>
#include <udontokenizers.h>

#define EXPECT(condition)                                                      \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "expectation failed at %s:%d: %s\n", __FILE__, __LINE__, \
              #condition);                                                     \
      return 1;                                                                \
    }                                                                          \
  } while (0)

/* A synthetic Llama-style model: three special added tokens, 256 byte-fallback
 * tokens at 3 + byte, and "U+2581", "a", "b", "ab", "U+2581ab" at 259..263. */
static int32_t decode_tokens(const udontokenizers_model_t *model,
                             const int32_t *tokens, int32_t count,
                             int32_t skip_special, int32_t *output,
                             int32_t capacity, int32_t *output_count) {
  udontokenizers_decoder_t decoder;
  int32_t index;
  udontokenizersStatus_t status = udontokenizersDecoderBeginRaw(
      model, &decoder, output, capacity, skip_special);
  for (index = 0; status == UDONTOKENIZERS_STATUS_SUCCESS && index < count;
       ++index)
    status =
        udontokenizersDecoderAppendRaw(model, &decoder, tokens[index], output);
  if (status == UDONTOKENIZERS_STATUS_SUCCESS)
    status =
        udontokenizersDecoderFinishRaw(model, &decoder, output, output_count);
  return status;
}

static int test_text_layer(void) {
  static const uint32_t scalars[] = {0x61, 0x62, 0x2581};
  static const int32_t scalar_tokens[] = {260, 261, 259};
  static const int32_t merge_left[] = {259, 260};
  static const int32_t merge_right[] = {262, 261};
  static const int32_t merge_result[] = {263, 262};
  static const int32_t merge_ranks[] = {1, 0};
  static const int32_t added_ids[] = {0, 2, 1};
  static const int32_t added_offsets[] = {0, 5, 9, 12};
  static const uint16_t added_units[] = {'<', 'u', 'n', 'k', '>', '<',
                                         '/', 's', '>', '<', 's', '>'};
  static const uint16_t piece_units[] = {'<', 'u', 'n', 'k', '>', '<', 's',
                                         '>', '<', '/', 's', '>', ' ', 'a',
                                         'b', 'a', 'b', ' ', 'a', 'b'};
  static const uint16_t chat_units[] = {'a', '<', '/', 's', '>', 'b'};
  static const int32_t chat_prefix_offsets[] = {0, 0, 0};
  static const int32_t chat_prefix_lengths[] = {-1, 1, -1};
  static const int32_t chat_suffix_offsets[] = {0, 1, 0};
  static const int32_t chat_suffix_lengths[] = {-1, 4, -1};
  static const uint16_t text_units[] = {'a', 'b', '<', '/', 's',
                                        '>', 'a', 'b', '<', 's'};
  static const uint16_t content_units[] = {'b'};
  int32_t byte_tokens[256];
  int32_t piece_offsets[265];
  int32_t piece_kinds[264];
  udontokenizers_model_t model;
  udontokenizers_chat_template_t chat;
  udontokenizers_encoder_t encoder;
  udon_string_native_t text = {text_units, 8};
  udon_string_native_t partial = {text_units, 10};
  udon_string_native_t content = {content_units, 1};
  int32_t tokens[16];
  int32_t units[16];
  int32_t count = -1;
  int32_t index;

  for (index = 0; index < 256; ++index) {
    byte_tokens[index] = 3 + index;
    piece_kinds[3 + index] = index;
  }
  piece_kinds[0] = piece_kinds[1] = piece_kinds[2] =
      UDONTOKENIZERS_PIECE_SPECIAL;
  for (index = 259; index < 264; ++index)
    piece_kinds[index] = UDONTOKENIZERS_PIECE_ORDINARY;
  piece_offsets[0] = 0;
  piece_offsets[1] = 5;
  piece_offsets[2] = 8;
  for (index = 3; index <= 259; ++index)
    piece_offsets[index] = 12;
  piece_offsets[260] = 13;
  piece_offsets[261] = 14;
  piece_offsets[262] = 15;
  piece_offsets[263] = 17;
  piece_offsets[264] = 20;

  model.scalar_values = scalars;
  model.scalar_token_ids = scalar_tokens;
  model.scalar_count = 3;
  model.merge_left_token_ids = merge_left;
  model.merge_right_token_ids = merge_right;
  model.merge_result_token_ids = merge_result;
  model.merge_ranks = merge_ranks;
  model.merge_count = 2;
  model.byte_token_ids = byte_tokens;
  model.normalizer_prepend_scalar = 0x2581;
  model.normalizer_replace_from_scalar = 0x20;
  model.normalizer_replace_to_scalar = 0x2581;
  model.bos_token_id = 1;
  model.added_token_ids = added_ids;
  model.added_token_offsets = added_offsets;
  model.added_token_code_units = added_units;
  model.added_token_count = 3;
  model.piece_offsets = piece_offsets;
  model.piece_code_units = piece_units;
  model.piece_kinds = piece_kinds;
  model.vocab_count = 264;
  model.decoder_strip_scalar = ' ';
  model.decoder_strip_count = 1;
  model.eos_token_id = 2;

  /* Added tokens split normalized segments; every segment gets the prefix. */
  EXPECT(udontokenizersEncodeRaw(&model, &text, tokens, 16, &count) ==
         UDONTOKENIZERS_STATUS_SUCCESS);
  EXPECT(count == 4);
  EXPECT(tokens[0] == 1 && tokens[1] == 263 && tokens[2] == 2 &&
         tokens[3] == 263);
  /* A partial added token is ordinary text. */
  EXPECT(udontokenizersEncodeRaw(&model, &partial, tokens, 16, &count) ==
         UDONTOKENIZERS_STATUS_SUCCESS);
  EXPECT(count == 6);
  EXPECT(tokens[3] == 263 && tokens[4] == 3 + '<' && tokens[5] == 3 + 's');

  /* Chat literals stream into one logical text: the literal "a" and content
   * "b" still merge across the piece boundary. */
  chat.code_units = chat_units;
  chat.role_prefix_offsets = chat_prefix_offsets;
  chat.role_prefix_lengths = chat_prefix_lengths;
  chat.role_suffix_offsets = chat_suffix_offsets;
  chat.role_suffix_lengths = chat_suffix_lengths;
  chat.generation_prompt_offset = 5;
  chat.generation_prompt_length = 1;
  EXPECT(udontokenizersEncoderBeginRaw(&model, &encoder, tokens, 16, 1) ==
         UDONTOKENIZERS_STATUS_SUCCESS);
  EXPECT(udontokenizersEncoderAppendChatMessageRaw(
             &model, &chat, &encoder, UDONTOKENIZERS_CHAT_ROLE_USER, &content,
             tokens) == UDONTOKENIZERS_STATUS_SUCCESS);
  EXPECT(udontokenizersEncoderAppendChatMessageRaw(
             &model, &chat, &encoder, UDONTOKENIZERS_CHAT_ROLE_SYSTEM, &content,
             tokens) == UDONTOKENIZERS_STATUS_INVALID_ARGUMENT);
  EXPECT(udontokenizersEncoderAppendChatGenerationPromptRaw(
             &model, &chat, &encoder, tokens) == UDONTOKENIZERS_STATUS_SUCCESS);
  EXPECT(udontokenizersEncoderFinishRaw(&model, &encoder, tokens, &count) ==
         UDONTOKENIZERS_STATUS_SUCCESS);
  EXPECT(count == 5);
  EXPECT(tokens[0] == 1 && tokens[1] == 263 && tokens[2] == 2 &&
         tokens[3] == 259 && tokens[4] == 261);
  EXPECT(udontokenizersEncoderBeginRaw(&model, &encoder, tokens, 2, 1) ==
         UDONTOKENIZERS_STATUS_SUCCESS);
  EXPECT(udontokenizersEncoderAppendCodeUnitsRaw(&model, &encoder, text_units,
                                                 8, tokens) ==
         UDONTOKENIZERS_STATUS_OUTPUT_TOO_SMALL);
  EXPECT(udontokenizersEncoderFinishRaw(&model, &encoder, tokens, &count) ==
         UDONTOKENIZERS_STATUS_OUTPUT_TOO_SMALL);

  /* Special tokens are skipped; the leading space is stripped once. */
  {
    static const int32_t ids[] = {1, 263, 2, 263};
    EXPECT(decode_tokens(&model, ids, 4, 1, units, 16, &count) ==
           UDONTOKENIZERS_STATUS_SUCCESS);
    EXPECT(count == 5);
    EXPECT(units[0] == 'a' && units[1] == 'b' && units[2] == ' ' &&
           units[3] == 'a' && units[4] == 'b');
    EXPECT(decode_tokens(&model, ids, 4, 0, units, 16, &count) ==
           UDONTOKENIZERS_STATUS_SUCCESS);
    EXPECT(count == 13);
    EXPECT(units[0] == '<' && units[3] == ' ' && units[4] == 'a');
  }
  /* A valid split UTF-8 run, U+1F35C as a surrogate pair. */
  {
    static const int32_t ids[] = {3 + 0xc3, 1,        3 + 0xa9, 3 + 0xf0,
                                  3 + 0x9f, 3 + 0x8d, 3 + 0x9c};
    EXPECT(decode_tokens(&model, ids, 7, 1, units, 16, &count) ==
           UDONTOKENIZERS_STATUS_SUCCESS);
    EXPECT(count == 3);
    EXPECT(units[0] == 0xe9 && units[1] == 0xd83c && units[2] == 0xdf5c);
  }
  /* An invalid run becomes one replacement per byte, including a leading
   * 0x20 byte that is therefore not stripped; a truncated run at the end of
   * input is invalid as well. */
  {
    static const int32_t ids[] = {3 + 0x20, 3 + 0xff, 260, 3 + 0xe3, 3 + 0x81};
    EXPECT(decode_tokens(&model, ids, 5, 1, units, 16, &count) ==
           UDONTOKENIZERS_STATUS_SUCCESS);
    EXPECT(count == 5);
    EXPECT(units[0] == 0xfffd && units[1] == 0xfffd && units[2] == 'a' &&
           units[3] == 0xfffd && units[4] == 0xfffd);
  }
  /* Overlong and surrogate encodings are invalid UTF-8. */
  {
    static const int32_t ids[] = {3 + 0xe0, 3 + 0x80, 3 + 0x80,
                                  3 + 0xed, 3 + 0xa0, 3 + 0x80};
    EXPECT(decode_tokens(&model, ids, 6, 1, units, 16, &count) ==
           UDONTOKENIZERS_STATUS_SUCCESS);
    EXPECT(count == 6);
    for (index = 0; index < 6; ++index)
      EXPECT(units[index] == 0xfffd);
  }
  {
    static const int32_t ids[] = {263, 264};
    EXPECT(decode_tokens(&model, ids, 2, 1, units, 16, &count) ==
           UDONTOKENIZERS_STATUS_INVALID_TOKEN);
    EXPECT(decode_tokens(&model, ids, 1, 1, units, 2, &count) ==
           UDONTOKENIZERS_STATUS_OUTPUT_TOO_SMALL);
  }
  return 0;
}

int main(void) {
  static const uint32_t scalars[] = {0x20, 0x65e5, 0x672c, 0x8a9e};
  static const int32_t scalar_tokens[] = {13, 10, 11, 12};
  static const int32_t merge_left_tokens[] = {10, 20};
  static const int32_t merge_right_tokens[] = {11, 12};
  static const int32_t merge_result_tokens[] = {20, 30};
  static const int32_t merge_ranks[] = {0, 1};
  int32_t byte_tokens[256];
  udontokenizers_model_t model;
  const uint16_t text_data[] = {0x65e5, 0x672c, 0x8a9e, 0x20, 0xd83c, 0xdf5c};
  udon_string_native_t text = {text_data, 6};
  int32_t token_data[8] = {0};
  int32_t raw_token_data[8] = {0};
  udon_int32_array_native_t tokens = {token_data, 8};
  int32_t token_count = -1;
  int32_t index;

  for (index = 0; index < 256; ++index)
    byte_tokens[index] = 1000 + index;
  model.scalar_values = scalars;
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

  EXPECT(udontokenizersEncode(&model, &text, &tokens, 8, &token_count) ==
         UDONTOKENIZERS_STATUS_SUCCESS);
  EXPECT(token_count == 6);
  EXPECT(token_data[0] == 30);
  EXPECT(token_data[1] == 13);
  EXPECT(token_data[2] == 1240);
  EXPECT(token_data[3] == 1159);
  EXPECT(token_data[4] == 1141);
  EXPECT(token_data[5] == 1156);

  token_count = -1;
  EXPECT(
      udontokenizersEncodeRaw(&model, &text, raw_token_data, 8, &token_count) ==
      UDONTOKENIZERS_STATUS_SUCCESS);
  EXPECT(token_count == 6);
  for (index = 0; index < token_count; ++index)
    EXPECT(raw_token_data[index] == token_data[index]);
  EXPECT(
      udontokenizersEncodeRaw(&model, &text, raw_token_data, 5, &token_count) ==
      UDONTOKENIZERS_STATUS_OUTPUT_TOO_SMALL);

  EXPECT(udontokenizersEncode(&model, &text, &tokens, 5, &token_count) ==
         UDONTOKENIZERS_STATUS_OUTPUT_TOO_SMALL);
  return test_text_layer();
}
