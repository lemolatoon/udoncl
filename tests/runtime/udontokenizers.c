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

#define SYMBOL_CAPACITY 64
#define WORKSPACE_WORDS UDONTOKENIZERS_ENCODER_WORKSPACE_WORDS(SYMBOL_CAPACITY)

static int32_t workspace[WORKSPACE_WORDS];

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
  model.merge_hash_slots = 0;
  model.merge_hash_mask = 0;
  model.merge_hash_multiplier = 0;

  /* Added tokens split normalized segments; every segment gets the prefix. */
  EXPECT(udontokenizersEncodeRaw(&model, &text, tokens, 16, &count, workspace,
                                 WORKSPACE_WORDS) ==
         UDONTOKENIZERS_STATUS_SUCCESS);
  EXPECT(count == 4);
  EXPECT(tokens[0] == 1 && tokens[1] == 263 && tokens[2] == 2 &&
         tokens[3] == 263);
  /* A partial added token is ordinary text. */
  EXPECT(udontokenizersEncodeRaw(&model, &partial, tokens, 16, &count,
                                 workspace, WORKSPACE_WORDS) ==
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
  EXPECT(udontokenizersEncoderBeginRaw(&model, &encoder, workspace,
                                       WORKSPACE_WORDS,
                                       1) == UDONTOKENIZERS_STATUS_SUCCESS);
  EXPECT(udontokenizersEncoderAppendChatMessageRaw(
             &model, &chat, &encoder, UDONTOKENIZERS_CHAT_ROLE_USER, &content,
             workspace) == UDONTOKENIZERS_STATUS_SUCCESS);
  EXPECT(udontokenizersEncoderAppendChatMessageRaw(
             &model, &chat, &encoder, UDONTOKENIZERS_CHAT_ROLE_SYSTEM, &content,
             workspace) == UDONTOKENIZERS_STATUS_INVALID_ARGUMENT);
  EXPECT(udontokenizersEncoderAppendChatGenerationPromptRaw(
             &model, &chat, &encoder, workspace) ==
         UDONTOKENIZERS_STATUS_SUCCESS);
  EXPECT(udontokenizersEncoderFinishRaw(&model, &encoder, workspace, &count) ==
         UDONTOKENIZERS_STATUS_SUCCESS);
  EXPECT(count == 5);
  EXPECT(workspace[0] == 1 && workspace[1] == 263 && workspace[2] == 2 &&
         workspace[3] == 259 && workspace[4] == 261);
  /* Text is rejected once merging has started. */
  EXPECT(udontokenizersEncoderAppendCodeUnitsRaw(&model, &encoder, text_units,
                                                 8, workspace) ==
         UDONTOKENIZERS_STATUS_INVALID_ARGUMENT);
  /* The same message in single-unit, single-work-unit slices. */
  {
    int32_t position = 0;
    int32_t finished = 0;
    int32_t calls = 0;
    EXPECT(udontokenizersEncoderBeginRaw(&model, &encoder, workspace,
                                         WORKSPACE_WORDS,
                                         1) == UDONTOKENIZERS_STATUS_SUCCESS);
    while (!finished) {
      EXPECT(udontokenizersEncoderAppendChatMessagePartRaw(
                 &model, &chat, &encoder, UDONTOKENIZERS_CHAT_ROLE_USER,
                 &content, &position, 1, &finished,
                 workspace) == UDONTOKENIZERS_STATUS_SUCCESS);
      ++calls;
    }
    /* "a", "b", then the whole added token "</s>" in one call. */
    EXPECT(calls == 3 && position == 6);
    EXPECT(udontokenizersEncoderAppendChatGenerationPromptRaw(
               &model, &chat, &encoder, workspace) ==
           UDONTOKENIZERS_STATUS_SUCCESS);
    finished = 0;
    calls = 0;
    while (!finished) {
      EXPECT(udontokenizersEncoderStepRaw(&model, &encoder, workspace, 1,
                                          &finished) ==
             UDONTOKENIZERS_STATUS_SUCCESS);
      ++calls;
    }
    EXPECT(calls > 5);
    EXPECT(
        udontokenizersEncoderFinishRaw(&model, &encoder, workspace, &count) ==
        UDONTOKENIZERS_STATUS_SUCCESS);
    EXPECT(count == 5);
    EXPECT(workspace[0] == 1 && workspace[1] == 263 && workspace[2] == 2 &&
           workspace[3] == 259 && workspace[4] == 261);
  }
  /* Room for two symbols: BOS and the prefix fit, the first letter does not,
   * and the failure is sticky. */
  EXPECT(
      udontokenizersEncoderBeginRaw(&model, &encoder, workspace,
                                    UDONTOKENIZERS_ENCODER_WORKSPACE_WORDS(2),
                                    1) == UDONTOKENIZERS_STATUS_SUCCESS);
  EXPECT(udontokenizersEncoderAppendCodeUnitsRaw(&model, &encoder, text_units,
                                                 8, workspace) ==
         UDONTOKENIZERS_STATUS_OUTPUT_TOO_SMALL);
  EXPECT(udontokenizersEncoderFinishRaw(&model, &encoder, workspace, &count) ==
         UDONTOKENIZERS_STATUS_OUTPUT_TOO_SMALL);
  /* The output only has to hold the merged tokens. */
  EXPECT(udontokenizersEncodeRaw(&model, &text, tokens, 4, &count, workspace,
                                 WORKSPACE_WORDS) ==
         UDONTOKENIZERS_STATUS_SUCCESS);
  EXPECT(udontokenizersEncodeRaw(&model, &text, tokens, 3, &count, workspace,
                                 WORKSPACE_WORDS) ==
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
  /* Streaming: final units are a growing prefix of the decoded text. The
   * leading strip is applied to the first final unit, an open byte-fallback
   * run is withheld, and a run that turns invalid is replaced as a whole. */
  {
    static const int32_t ids[] = {1,        263,      3 + 0xe3, 3 + 0x81,
                                  3 + 0x82, 3 + 0xc3, 2,        3 + 0xa9,
                                  260,      3 + 0xe3, 3 + 0x81, 3 + 0xff,
                                  261,      3 + 0xf0, 3 + 0x9f};
    static const int32_t chunk_lengths[] = {0, 2, 0, 0, 0, 0, 0, 0,
                                            3, 0, 0, 0, 4, 0, 0};
    static const int32_t expected[] = {'a', 'b',    0x3042, 0xe9,
                                       'a', 0xfffd, 0xfffd, 0xfffd,
                                       'b', 0xfffd, 0xfffd};
    udontokenizers_decoder_t decoder;
    int32_t streamed = 0;
    int32_t offset = -1;
    int32_t length = -1;
    EXPECT(udontokenizersDecoderBeginRaw(&model, &decoder, units, 16, 1) ==
           UDONTOKENIZERS_STATUS_SUCCESS);
    for (index = 0; index < 15; ++index) {
      EXPECT(udontokenizersDecoderStepRaw(&model, &decoder, ids[index], units,
                                          &offset, &length) ==
             UDONTOKENIZERS_STATUS_SUCCESS);
      EXPECT(offset == streamed && length == chunk_lengths[index]);
      streamed += length;
      EXPECT(decoder.final_count == streamed);
      for (count = 0; count < streamed; ++count)
        EXPECT(units[count] == expected[count]);
    }
    EXPECT(streamed == 9);
    EXPECT(udontokenizersDecoderFinishRaw(&model, &decoder, units, &count) ==
           UDONTOKENIZERS_STATUS_SUCCESS);
    EXPECT(count == 11 && decoder.final_count == 11);
    for (index = 0; index < 11; ++index)
      EXPECT(units[index] == expected[index]);
    EXPECT(udontokenizersDecoderStepRaw(&model, &decoder, 264, units, &offset,
                                        &length) ==
           UDONTOKENIZERS_STATUS_INVALID_TOKEN);
    EXPECT(offset == 11 && length == 0);
  }
  /* A strip count of two removes leading spaces across pieces, including a
   * byte-fallback space, but never a replacement character. */
  {
    static const int32_t spaces[] = {259, 3 + 0x20, 259, 260};
    static const int32_t invalid[] = {3 + 0x20, 3 + 0xff, 259};
    model.decoder_strip_count = 2;
    EXPECT(decode_tokens(&model, spaces, 4, 1, units, 16, &count) ==
           UDONTOKENIZERS_STATUS_SUCCESS);
    EXPECT(count == 2 && units[0] == ' ' && units[1] == 'a');
    EXPECT(decode_tokens(&model, invalid, 3, 1, units, 16, &count) ==
           UDONTOKENIZERS_STATUS_SUCCESS);
    EXPECT(count == 3 && units[0] == 0xfffd && units[1] == 0xfffd &&
           units[2] == ' ');
    model.decoder_strip_count = 1;
  }
  return 0;
}

/* Reference BPE: repeatedly merge the lowest-ranked pair, leftmost first,
 * by scanning every adjacent pair. This is the definition the queue-based
 * encoder must reproduce for any merge table. */
static int32_t reference_merge(const int32_t *lefts, const int32_t *rights,
                               const int32_t *results, const int32_t *ranks,
                               int32_t merge_count, int32_t *symbols,
                               int32_t count) {
  for (;;) {
    int32_t best = -1;
    int32_t best_index = -1;
    int32_t index;
    int32_t merge;
    for (index = 0; index + 1 < count; ++index)
      for (merge = 0; merge < merge_count; ++merge)
        if (lefts[merge] == symbols[index] &&
            rights[merge] == symbols[index + 1] &&
            (best < 0 || ranks[merge] < ranks[best])) {
          best = merge;
          best_index = index;
        }
    if (best < 0)
      return count;
    symbols[best_index] = results[best];
    for (index = best_index + 1; index + 1 < count; ++index)
      symbols[index] = symbols[index + 1];
    --count;
  }
}

static uint32_t random_state = 0x9e3779b9u;

static int32_t random_below(int32_t bound) {
  random_state = random_state * 1664525u + 1013904223u;
  return (int32_t)((random_state >> 8) % (uint32_t)bound);
}

/* Random merge tables over a three-letter alphabet, including merges whose
 * result is an existing token, and random texts with an added token that
 * separates segments. */
static int test_merge_order(void) {
  enum { MERGES = 12, LENGTH = 48, BASE = 3, BARRIER = 50, SLOTS = 32 };
  static const uint32_t scalars[] = {'a', 'b', 'c'};
  static const int32_t scalar_tokens[] = {0, 1, 2};
  static const int32_t added_ids[] = {BARRIER};
  static const int32_t added_offsets[] = {0, 1};
  static const uint16_t added_units[] = {'|'};
  static int32_t large[UDONTOKENIZERS_ENCODER_WORKSPACE_WORDS(LENGTH)];
  int32_t byte_tokens[256];
  int32_t lefts[MERGES];
  int32_t rights[MERGES];
  int32_t results[MERGES];
  int32_t ranks[MERGES];
  int32_t slots[SLOTS];
  uint16_t text[LENGTH];
  int32_t expected[LENGTH];
  udontokenizers_model_t model;
  udontokenizers_encoder_t encoder;
  int32_t round;
  int32_t index;

  for (index = 0; index < 256; ++index)
    byte_tokens[index] = 100 + index;
  model.scalar_values = scalars;
  model.scalar_token_ids = scalar_tokens;
  model.scalar_count = 3;
  model.merge_left_token_ids = lefts;
  model.merge_right_token_ids = rights;
  model.merge_result_token_ids = results;
  model.merge_ranks = ranks;
  model.byte_token_ids = byte_tokens;
  model.normalizer_prepend_scalar = -1;
  model.normalizer_replace_from_scalar = -1;
  model.normalizer_replace_to_scalar = -1;
  model.bos_token_id = -1;
  model.added_token_ids = added_ids;
  model.added_token_offsets = added_offsets;
  model.added_token_code_units = added_units;
  model.added_token_count = 1;
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

  for (round = 0; round < 3000; ++round) {
    int32_t merge_count = random_below(MERGES + 1);
    int32_t length = random_below(LENGTH + 1);
    int32_t tokens = BASE;
    int32_t expected_count = 0;
    int32_t segment = 0;
    int32_t count = -1;
    int32_t position = 0;
    int32_t finished = 0;
    int32_t merge;
    /* Distinct pairs in ascending (left, right) order, random results. */
    for (merge = 0; merge < merge_count;) {
      int32_t left = random_below(tokens);
      int32_t right = random_below(tokens);
      int32_t insert;
      if (merge >= tokens * tokens) {
        merge_count = merge;
        break;
      }
      for (insert = 0; insert < merge; ++insert)
        if (lefts[insert] == left && rights[insert] == right)
          break;
      if (insert < merge)
        continue;
      for (insert = merge; insert > 0 && (lefts[insert - 1] > left ||
                                          (lefts[insert - 1] == left &&
                                           rights[insert - 1] > right));
           --insert) {
        lefts[insert] = lefts[insert - 1];
        rights[insert] = rights[insert - 1];
        results[insert] = results[insert - 1];
      }
      lefts[insert] = left;
      rights[insert] = right;
      results[insert] = random_below(4) ? tokens++ : random_below(tokens);
      ++merge;
    }
    for (merge = 0; merge < merge_count; ++merge)
      ranks[merge] = merge;
    for (merge = merge_count - 1; merge > 0; --merge) {
      int32_t other = random_below(merge + 1);
      int32_t rank = ranks[merge];
      ranks[merge] = ranks[other];
      ranks[other] = rank;
    }
    model.merge_count = merge_count;
    /* Half of the rounds look merges up through the hash index. */
    model.merge_hash_slots = 0;
    if (round & 2) {
      for (index = 0; index < SLOTS; ++index)
        slots[index] = -1;
      for (merge = 0; merge < merge_count; ++merge) {
        int32_t slot = (lefts[merge] * 7 + rights[merge]) & (SLOTS - 1);
        while (slots[slot] >= 0)
          slot = (slot + 1) & (SLOTS - 1);
        slots[slot] = merge;
      }
      model.merge_hash_slots = slots;
      model.merge_hash_mask = SLOTS - 1;
      model.merge_hash_multiplier = 7;
    }
    for (index = 0; index < length; ++index)
      text[index] =
          random_below(9) ? (uint16_t)('a' + random_below(3)) : (uint16_t)'|';
    for (index = 0; index <= length; ++index) {
      if (index < length && text[index] != '|') {
        expected[expected_count++] = text[index] - 'a';
        continue;
      }
      expected_count =
          segment + reference_merge(lefts, rights, results, ranks, merge_count,
                                    expected + segment,
                                    expected_count - segment);
      if (index < length)
        expected[expected_count++] = BARRIER;
      segment = expected_count;
    }

    EXPECT(udontokenizersEncoderBeginRaw(
               &model, &encoder, large,
               UDONTOKENIZERS_ENCODER_WORKSPACE_WORDS(LENGTH),
               0) == UDONTOKENIZERS_STATUS_SUCCESS);
    if (round & 1) {
      /* Bounded: three code units and two work units per call. */
      udon_string_native_t string = {text, length};
      while (!finished)
        EXPECT(udontokenizersEncoderAppendPartRaw(
                   &model, &encoder, &string, &position, 3, &finished, large) ==
               UDONTOKENIZERS_STATUS_SUCCESS);
      EXPECT(position == length);
      finished = 0;
      while (!finished)
        EXPECT(udontokenizersEncoderStepRaw(&model, &encoder, large, 2,
                                            &finished) ==
               UDONTOKENIZERS_STATUS_SUCCESS);
    } else {
      EXPECT(udontokenizersEncoderAppendCodeUnitsRaw(&model, &encoder, text,
                                                     length, large) ==
             UDONTOKENIZERS_STATUS_SUCCESS);
    }
    EXPECT(udontokenizersEncoderFinishRaw(&model, &encoder, large, &count) ==
           UDONTOKENIZERS_STATUS_SUCCESS);
    EXPECT(count == expected_count);
    for (index = 0; index < count; ++index)
      EXPECT(large[index] == expected[index]);
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
  model.merge_hash_slots = 0;
  model.merge_hash_mask = 0;
  model.merge_hash_multiplier = 0;

  EXPECT(udontokenizersEncode(&model, &text, &tokens, 8, &token_count,
                              workspace, WORKSPACE_WORDS) ==
         UDONTOKENIZERS_STATUS_SUCCESS);
  EXPECT(token_count == 6);
  EXPECT(token_data[0] == 30);
  EXPECT(token_data[1] == 13);
  EXPECT(token_data[2] == 1240);
  EXPECT(token_data[3] == 1159);
  EXPECT(token_data[4] == 1141);
  EXPECT(token_data[5] == 1156);

  token_count = -1;
  EXPECT(udontokenizersEncodeRaw(&model, &text, raw_token_data, 8, &token_count,
                                 workspace, WORKSPACE_WORDS) ==
         UDONTOKENIZERS_STATUS_SUCCESS);
  EXPECT(token_count == 6);
  for (index = 0; index < token_count; ++index)
    EXPECT(raw_token_data[index] == token_data[index]);
  EXPECT(udontokenizersEncodeRaw(&model, &text, raw_token_data, 5, &token_count,
                                 workspace, WORKSPACE_WORDS) ==
         UDONTOKENIZERS_STATUS_OUTPUT_TOO_SMALL);
  EXPECT(udontokenizersEncode(&model, &text, &tokens, 5, &token_count,
                              workspace, WORKSPACE_WORDS) ==
         UDONTOKENIZERS_STATUS_OUTPUT_TOO_SMALL);
  /* Eight symbols are needed before merging; seven are not enough. */
  EXPECT(udontokenizersEncodeRaw(&model, &text, raw_token_data, 8, &token_count,
                                 workspace,
                                 UDONTOKENIZERS_ENCODER_WORKSPACE_WORDS(7)) ==
         UDONTOKENIZERS_STATUS_OUTPUT_TOO_SMALL);
  EXPECT(udontokenizersEncodeRaw(&model, &text, raw_token_data, 8, &token_count,
                                 workspace,
                                 UDONTOKENIZERS_ENCODER_WORKSPACE_WORDS(8)) ==
         UDONTOKENIZERS_STATUS_SUCCESS);
  EXPECT(token_count == 6);
  if (test_merge_order())
    return 1;
  return test_text_layer();
}
