#ifndef UDONTOKENIZERS_H
#define UDONTOKENIZERS_H

#include <stdint.h>
#include <udoncl/array.h>
#include <udoncl/string.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum udontokenizersStatus {
  UDONTOKENIZERS_STATUS_SUCCESS = 0,
  UDONTOKENIZERS_STATUS_INVALID_ARGUMENT = 1,
  UDONTOKENIZERS_STATUS_INVALID_MODEL = 2,
  UDONTOKENIZERS_STATUS_INVALID_UTF16 = 3,
  UDONTOKENIZERS_STATUS_OUTPUT_TOO_SMALL = 4,
  UDONTOKENIZERS_STATUS_INVALID_TOKEN = 5
} udontokenizersStatus_t;

/* Decoder piece kinds; non-negative kinds are byte-fallback byte values. */
#define UDONTOKENIZERS_PIECE_ORDINARY (-1)
#define UDONTOKENIZERS_PIECE_SPECIAL (-2)

typedef struct udontokenizers_model {
  /* scalar_values and merges are sorted for deterministic binary lookup. */
  const uint32_t *scalar_values;
  const int32_t *scalar_token_ids;
  int32_t scalar_count;
  const int32_t *merge_left_token_ids;
  const int32_t *merge_right_token_ids;
  const int32_t *merge_result_token_ids;
  const int32_t *merge_ranks;
  int32_t merge_count;
  const int32_t *byte_token_ids;
  int32_t normalizer_prepend_scalar;
  int32_t normalizer_replace_from_scalar;
  int32_t normalizer_replace_to_scalar;
  int32_t bos_token_id;
  /* Added tokens are matched on the raw UTF-16 input before normalization,
   * leftmost first and then longest. Entries are ordered by descending
   * code-unit length, so the first match at a position is the longest. Text
   * between matches forms independent normalized BPE segments. A zero count
   * disables matching. */
  const int32_t *added_token_ids;
  const int32_t *added_token_offsets; /* added_token_count + 1 entries */
  const uint16_t *added_token_code_units;
  int32_t added_token_count;
  /* Decoder tables, indexed by token ID. Pieces are UTF-16 after the
   * decoder's scalar replacement. Kinds are a byte value for byte-fallback
   * pieces or one of the UDONTOKENIZERS_PIECE_* values. A zero vocab_count
   * disables decoding. */
  const int32_t *piece_offsets; /* vocab_count + 1 entries */
  const uint16_t *piece_code_units;
  const int32_t *piece_kinds;
  int32_t vocab_count;
  /* Remove up to decoder_strip_count leading decoder_strip_scalar code units
   * from the fused decoded text; -1 disables. */
  int32_t decoder_strip_scalar;
  int32_t decoder_strip_count;
  int32_t eos_token_id;
} udontokenizers_model_t;

/* One-shot encode: optional BOS, added-token splitting, normalization and
 * byte-fallback BPE, matching Hugging Face `Tokenizer.encode(text)`. */
udontokenizersStatus_t udontokenizersEncode(const udontokenizers_model_t *model,
                                            udon_string_t input,
                                            udon_int32_array_t output,
                                            int32_t output_capacity,
                                            int32_t *output_count);

udontokenizersStatus_t
udontokenizersEncodeRaw(const udontokenizers_model_t *model,
                        udon_string_t input, int32_t *output,
                        int32_t output_capacity, int32_t *output_count);

/* Streaming encode of one logical text supplied as consecutive pieces. BPE
 * segments continue across pieces and end only at added tokens; an added
 * token must lie within one piece. The caller owns the state and output. */
typedef struct udontokenizers_encoder {
  int32_t count;
  int32_t capacity;
  int32_t segment_first;
  int32_t segment_started;
  int32_t status;
} udontokenizers_encoder_t;

udontokenizersStatus_t udontokenizersEncoderBeginRaw(
    const udontokenizers_model_t *model, udontokenizers_encoder_t *encoder,
    int32_t *output, int32_t output_capacity, int32_t add_special_tokens);
udontokenizersStatus_t
udontokenizersEncoderAppendRaw(const udontokenizers_model_t *model,
                               udontokenizers_encoder_t *encoder,
                               udon_string_t text, int32_t *output);
udontokenizersStatus_t udontokenizersEncoderAppendCodeUnitsRaw(
    const udontokenizers_model_t *model, udontokenizers_encoder_t *encoder,
    const uint16_t *code_units, int32_t length, int32_t *output);
udontokenizersStatus_t
udontokenizersEncoderFinishRaw(const udontokenizers_model_t *model,
                               udontokenizers_encoder_t *encoder,
                               int32_t *output, int32_t *output_count);

/* A chat template compiled at build time into per-role literal prefixes and
 * suffixes around the message content plus a generation prompt. Lengths of
 * -1 mark roles the template does not render. */
#define UDONTOKENIZERS_CHAT_ROLE_SYSTEM 0
#define UDONTOKENIZERS_CHAT_ROLE_USER 1
#define UDONTOKENIZERS_CHAT_ROLE_ASSISTANT 2
#define UDONTOKENIZERS_CHAT_ROLE_COUNT 3

typedef struct udontokenizers_chat_template {
  const uint16_t *code_units;
  const int32_t *role_prefix_offsets; /* UDONTOKENIZERS_CHAT_ROLE_COUNT */
  const int32_t *role_prefix_lengths;
  const int32_t *role_suffix_offsets;
  const int32_t *role_suffix_lengths;
  int32_t generation_prompt_offset;
  int32_t generation_prompt_length;
} udontokenizers_chat_template_t;

udontokenizersStatus_t udontokenizersEncoderAppendChatMessageRaw(
    const udontokenizers_model_t *model,
    const udontokenizers_chat_template_t *chat,
    udontokenizers_encoder_t *encoder, int32_t role, udon_string_t content,
    int32_t *output);
udontokenizersStatus_t udontokenizersEncoderAppendChatGenerationPromptRaw(
    const udontokenizers_model_t *model,
    const udontokenizers_chat_template_t *chat,
    udontokenizers_encoder_t *encoder, int32_t *output);

/* Incremental decode to UTF-16 code units (one per int32 element), matching
 * Hugging Face `decode` for Replace/ByteFallback/Fuse/Strip decoders. A run of
 * byte-fallback pieces that is not valid UTF-8 becomes one U+FFFD per byte;
 * the leading strip is applied by Finish. */
typedef struct udontokenizers_decoder {
  int32_t count;
  int32_t capacity;
  int32_t skip_special_tokens;
  int32_t run_start;
  int32_t run_bytes;
  int32_t run_invalid;
  int32_t pending_scalar;
  int32_t pending_remaining;
  int32_t pending_lower;
  int32_t pending_upper;
  int32_t status;
} udontokenizers_decoder_t;

udontokenizersStatus_t udontokenizersDecoderBeginRaw(
    const udontokenizers_model_t *model, udontokenizers_decoder_t *decoder,
    int32_t *output, int32_t output_capacity, int32_t skip_special_tokens);
udontokenizersStatus_t
udontokenizersDecoderAppendRaw(const udontokenizers_model_t *model,
                               udontokenizers_decoder_t *decoder,
                               int32_t token_id, int32_t *output);
udontokenizersStatus_t
udontokenizersDecoderFinishRaw(const udontokenizers_model_t *model,
                               udontokenizers_decoder_t *decoder,
                               int32_t *output, int32_t *output_count);

#ifdef __cplusplus
}
#endif

#endif
