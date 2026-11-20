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
  /* merge_ranks is a permutation of 0..merge_count-1. */
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
  /* Optional open-addressing index of the merge table; null selects binary
   * search. The pair (left, right) starts probing at slot
   * (left * merge_hash_multiplier + right) & merge_hash_mask and advances by
   * one slot. A slot holds a merge table index or -1 when empty, at least
   * one slot is empty, and the multiplication does not overflow int32. */
  const int32_t *merge_hash_slots;
  int32_t merge_hash_mask;
  int32_t merge_hash_multiplier;
} udontokenizers_model_t;

/* Encoder workspace: caller-owned int32 words used while one text is encoded.
 * A workspace of UDONTOKENIZERS_ENCODER_WORKSPACE_WORDS(n) words holds a text
 * of at most n pre-merge symbols: one per character with a vocabulary entry,
 * one per UTF-8 byte of a byte-fallback character, one per added token and
 * one per normalizer prefix. A longer text fails with OUTPUT_TOO_SMALL.
 *
 * Merging follows the Hugging Face BPE order exactly: a priority queue of
 * adjacent pairs keyed by (merge rank, position) over a doubly linked symbol
 * list, at O(n log n) cost. The queue packs rank and position into one word,
 * which caps n just below 2^31 divided by the power of two above
 * merge_count. */
#define UDONTOKENIZERS_ENCODER_WORKSPACE_WORDS(symbol_capacity)                \
  ((symbol_capacity) * 6)

/* One-shot encode: optional BOS, added-token splitting, normalization and
 * byte-fallback BPE, matching Hugging Face `Tokenizer.encode(text)`. The
 * token IDs are copied to `output`. */
udontokenizersStatus_t
udontokenizersEncode(const udontokenizers_model_t *model, udon_string_t input,
                     udon_int32_array_t output, int32_t output_capacity,
                     int32_t *output_count, int32_t *workspace,
                     int32_t workspace_words);

udontokenizersStatus_t
udontokenizersEncodeRaw(const udontokenizers_model_t *model,
                        udon_string_t input, int32_t *output,
                        int32_t output_capacity, int32_t *output_count,
                        int32_t *workspace, int32_t workspace_words);

/* Streaming encode of one logical text supplied as consecutive pieces. BPE
 * segments continue across pieces and end only at added tokens; an added
 * token must lie within one piece. The caller owns the state and the
 * workspace and passes the same workspace to every call of one encode.
 *
 * Appending only converts text to symbols. Merging runs in Step or Finish;
 * once Finish succeeds or Step reports *finished, the token IDs are the first
 * `count` words of the workspace.
 *
 * Bounded work: every `...PartRaw` function and StepRaw does at most the
 * requested amount of work per call and keeps its progress in the encoder,
 * so a caller with a per-frame budget can spread one encode over many calls.
 * The unbounded functions equal the bounded ones called to completion. */
typedef struct udontokenizers_encoder {
  int32_t count; /* symbols while appending, tokens once finished */
  int32_t capacity;
  int32_t segment_started;
  int32_t phase;
  int32_t cursor;
  int32_t heap_count;
  int32_t position_bits;
  int32_t position_mask;
  int32_t status;
} udontokenizers_encoder_t;

udontokenizersStatus_t udontokenizersEncoderBeginRaw(
    const udontokenizers_model_t *model, udontokenizers_encoder_t *encoder,
    int32_t *workspace, int32_t workspace_words, int32_t add_special_tokens);
udontokenizersStatus_t
udontokenizersEncoderAppendRaw(const udontokenizers_model_t *model,
                               udontokenizers_encoder_t *encoder,
                               udon_string_t text, int32_t *workspace);
/* Appends `text` from code unit *position and stops at the first character
 * or added-token boundary that is at least `max_code_units` code units
 * later. *position is advanced and *finished is set once the whole text was
 * consumed. The caller passes the same text on every call. */
udontokenizersStatus_t udontokenizersEncoderAppendPartRaw(
    const udontokenizers_model_t *model, udontokenizers_encoder_t *encoder,
    udon_string_t text, int32_t *position, int32_t max_code_units,
    int32_t *finished, int32_t *workspace);
udontokenizersStatus_t udontokenizersEncoderAppendCodeUnitsRaw(
    const udontokenizers_model_t *model, udontokenizers_encoder_t *encoder,
    const uint16_t *code_units, int32_t length, int32_t *workspace);
/* Does at most `max_work` units of merge work. One unit links one symbol,
 * resolves one queued pair, or emits one token. */
udontokenizersStatus_t udontokenizersEncoderStepRaw(
    const udontokenizers_model_t *model, udontokenizers_encoder_t *encoder,
    int32_t *workspace, int32_t max_work, int32_t *finished);
udontokenizersStatus_t
udontokenizersEncoderFinishRaw(const udontokenizers_model_t *model,
                               udontokenizers_encoder_t *encoder,
                               int32_t *workspace, int32_t *output_count);

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
    int32_t *workspace);
/* Bounded form of AppendChatMessageRaw. *position counts the code units of
 * the role prefix, the content and the role suffix consumed so far. It starts
 * at zero for each message and advances by about `max_code_units` per call
 * until *finished is set. */
udontokenizersStatus_t udontokenizersEncoderAppendChatMessagePartRaw(
    const udontokenizers_model_t *model,
    const udontokenizers_chat_template_t *chat,
    udontokenizers_encoder_t *encoder, int32_t role, udon_string_t content,
    int32_t *position, int32_t max_code_units, int32_t *finished,
    int32_t *workspace);
/* Continues a conversation after a message of `previous_role` that the model
 * generated as token IDs ending with its end-of-sequence token, so the
 * generated tokens enter the context as they are instead of being decoded and
 * encoded again. Appends the rest of `previous_role`'s suffix after the
 * end-of-sequence text, then the `role` message as AppendChatMessageRaw does.
 * The suffix must start with the text of the model's end-of-sequence added
 * token. That token ends a segment, so the appended text starts a new
 * segment; an encoder begun without special tokens for this text yields the
 * same tokens as encoding the whole conversation in one text.
 *
 * For TinyLlama, appending a user turn after an assistant reply that ended
 * with `</s>` encodes "\n<|user|>\n{content}</s>\n". */
udontokenizersStatus_t udontokenizersEncoderAppendChatTurnRaw(
    const udontokenizers_model_t *model,
    const udontokenizers_chat_template_t *chat,
    udontokenizers_encoder_t *encoder, int32_t previous_role, int32_t role,
    udon_string_t content, int32_t *workspace);
/* Bounded form of AppendChatTurnRaw. *position counts the code units of the
 * rest of the previous suffix, the role prefix, the content and the role
 * suffix consumed so far, starting at zero for each turn. */
udontokenizersStatus_t udontokenizersEncoderAppendChatTurnPartRaw(
    const udontokenizers_model_t *model,
    const udontokenizers_chat_template_t *chat,
    udontokenizers_encoder_t *encoder, int32_t previous_role, int32_t role,
    udon_string_t content, int32_t *position, int32_t max_code_units,
    int32_t *finished, int32_t *workspace);
udontokenizersStatus_t udontokenizersEncoderAppendChatGenerationPromptRaw(
    const udontokenizers_model_t *model,
    const udontokenizers_chat_template_t *chat,
    udontokenizers_encoder_t *encoder, int32_t *workspace);

/* Incremental decode to UTF-16 code units (one per int32 element), matching
 * Hugging Face `decode` for Replace/ByteFallback/Fuse/Strip decoders. A run of
 * byte-fallback pieces that is not valid UTF-8 becomes one U+FFFD per byte.
 *
 * Streaming: the first `final_count` output units are final. The leading
 * strip is already applied to them and they never change, so they are always
 * a prefix of the text Finish produces. The units of an open byte-fallback
 * run are withheld until the run ends, because Hugging Face replaces the
 * whole run when any part of it is invalid UTF-8.
 *
 * StepRaw is the counterpart of Hugging Face `DecodeStream.step`: it appends
 * one token and reports the units that became final as
 * output[*chunk_offset .. *chunk_offset + *chunk_length). A zero length
 * corresponds to `step` returning None. Unlike `DecodeStream`, the chunks of
 * all steps followed by the units Finish adds always concatenate to exactly
 * `decode(ids)`, including for invalid or truncated UTF-8. Each call costs
 * time proportional to its own token only. */
typedef struct udontokenizers_decoder {
  int32_t count; /* units written, including an open byte-fallback run */
  int32_t capacity;
  int32_t skip_special_tokens;
  int32_t final_count;
  int32_t strip_pending;
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
udontokenizersDecoderStepRaw(const udontokenizers_model_t *model,
                             udontokenizers_decoder_t *decoder,
                             int32_t token_id, int32_t *output,
                             int32_t *chunk_offset, int32_t *chunk_length);
/* Ends the text: a still-open byte-fallback run is resolved and every unit
 * becomes final. *output_count is the length of the whole decoded text. */
udontokenizersStatus_t
udontokenizersDecoderFinishRaw(const udontokenizers_model_t *model,
                               udontokenizers_decoder_t *decoder,
                               int32_t *output, int32_t *output_count);

#ifdef __cplusplus
}
#endif

#endif
