#include <udontokenizers.h>

#define UDONTOKENIZERS_REPLACEMENT_CHARACTER 0xfffd
#define UDONTOKENIZERS_UNBOUNDED 0x7fffffff

/* Encoder phases. Text is accepted only in the first one. */
#define UDONTOKENIZERS_PHASE_TEXT 0
#define UDONTOKENIZERS_PHASE_LINK 1
#define UDONTOKENIZERS_PHASE_MERGE 2
#define UDONTOKENIZERS_PHASE_EMIT 3
#define UDONTOKENIZERS_PHASE_FINISHED 4

/* Workspace layout for a symbol capacity of n words per array:
 *   [0, n)   symbols: a token ID, or -1 - ID for an added token, which never
 *            merges and therefore separates BPE segments
 *   [n, 2n)  previous live symbol, -1 at the start
 *   [2n, 3n) next live symbol, -1 at the end
 *   [3n, 4n) merge table index of the pair starting at a symbol, or -1
 *   [4n, 6n) binary min-heap of (rank << position_bits) | position
 * A heap entry is stale when its symbol no longer starts a pair of that rank.
 * Every live mergeable pair always has an entry, so the smallest valid entry
 * is the lowest-ranked, leftmost pair: the pair Hugging Face merges next. */

static int32_t scalar_token(const udontokenizers_model_t *model,
                            uint32_t scalar) {
  /* Unicode scalars are below 2^31, so signed comparison is exact. */
  const int32_t *values = (const int32_t *)model->scalar_values;
  int32_t wanted = (int32_t)scalar;
  int32_t count = model->scalar_count;
  int32_t begin = 0;
  int32_t end = count;
  while (begin != end) {
    int32_t middle = begin + ((end - begin) >> 1);
    if (values[middle] < wanted)
      begin = middle + 1;
    else
      end = middle;
  }
  if (begin < count && values[begin] == wanted)
    return model->scalar_token_ids[begin];
  return -1;
}

static udontokenizersStatus_t append_symbol(udontokenizers_encoder_t *encoder,
                                            int32_t *workspace,
                                            int32_t symbol) {
  int32_t count = encoder->count;
  if (count >= encoder->capacity)
    return UDONTOKENIZERS_STATUS_OUTPUT_TOO_SMALL;
  workspace[count] = symbol;
  encoder->count = count + 1;
  return UDONTOKENIZERS_STATUS_SUCCESS;
}

static udontokenizersStatus_t
append_fallback(const udontokenizers_model_t *model,
                udontokenizers_encoder_t *encoder, int32_t *workspace,
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
        append_symbol(encoder, workspace, model->byte_token_ids[bytes[index]]);
    if (status != UDONTOKENIZERS_STATUS_SUCCESS)
      return status;
  }
  return UDONTOKENIZERS_STATUS_SUCCESS;
}

static udontokenizersStatus_t append_scalar(const udontokenizers_model_t *model,
                                            udontokenizers_encoder_t *encoder,
                                            int32_t *workspace,
                                            uint32_t scalar) {
  int32_t token = scalar_token(model, scalar);
  return token >= 0 ? append_symbol(encoder, workspace, token)
                    : append_fallback(model, encoder, workspace, scalar);
}

static int32_t find_merge(const int32_t *lefts, const int32_t *rights,
                          int32_t merge_count, const int32_t *slots,
                          int32_t mask, int32_t multiplier, int32_t left,
                          int32_t right) {
  int32_t begin = 0;
  int32_t end = merge_count;
  if (slots) {
    int32_t slot =
        (int32_t)(((uint32_t)left * (uint32_t)multiplier + (uint32_t)right) &
                  (uint32_t)mask);
    int32_t probes;
    for (probes = 0; probes <= mask; ++probes) {
      int32_t merge = slots[slot];
      if (merge < 0)
        return -1;
      if (lefts[merge] == left && rights[merge] == right)
        return merge;
      slot = (slot + 1) & mask;
    }
    return -1;
  }
  /* `begin != end` keeps the halving an arithmetic shift, which costs one
   * Udon instruction sequence instead of an emulated logical shift. */
  while (begin != end) {
    int32_t middle = begin + ((end - begin) >> 1);
    int32_t candidate_left = lefts[middle];
    if (candidate_left < left ||
        (candidate_left == left && rights[middle] < right))
      begin = middle + 1;
    else
      end = middle;
  }
  if (begin < merge_count && lefts[begin] == left && rights[begin] == right)
    return begin;
  return -1;
}

static int32_t valid_model(const udontokenizers_model_t *model) {
  if (model->scalar_count < 0 || model->merge_count < 0 ||
      !model->scalar_values || !model->scalar_token_ids ||
      (model->merge_count &&
       (!model->merge_left_token_ids || !model->merge_right_token_ids ||
        !model->merge_result_token_ids || !model->merge_ranks)) ||
      (model->merge_hash_slots &&
       (model->merge_hash_mask < 0 || model->merge_hash_multiplier <= 0 ||
        (model->merge_hash_mask & (model->merge_hash_mask + 1)) != 0)) ||
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

/* Returns the index of the longest added token starting at `position`, whose
 * code unit is `first`, or -1. */
static int32_t match_added_token(const udontokenizers_model_t *model,
                                 udon_string_t string, const uint16_t *units,
                                 int32_t use_units, int32_t length,
                                 int32_t position, uint32_t first) {
  const int32_t *offsets = model->added_token_offsets;
  const uint16_t *code_units = model->added_token_code_units;
  int32_t token_count = model->added_token_count;
  int32_t available = length - position;
  int32_t token;
  for (token = 0; token < token_count; ++token) {
    int32_t begin = offsets[token];
    int32_t token_length = offsets[token + 1] - begin;
    int32_t index;
    if (token_length <= 0 || token_length > available ||
        (uint32_t)code_units[begin] != first)
      continue;
    for (index = 1; index < token_length; ++index)
      if (source_unit(string, units, use_units, position + index) !=
          (uint32_t)code_units[begin + index])
        break;
    if (index == token_length)
      return token;
  }
  return -1;
}

static udontokenizersStatus_t fail(udontokenizers_encoder_t *encoder,
                                   udontokenizersStatus_t status) {
  encoder->status = status;
  return status;
}

/* Converts text to symbols from *position until the source ends or at least
 * `max_code_units` code units were consumed. */
static udontokenizersStatus_t
encoder_append(const udontokenizers_model_t *model,
               udontokenizers_encoder_t *encoder, udon_string_t string,
               const uint16_t *units, int32_t units_length, int32_t use_units,
               int32_t *workspace, int32_t *position_io,
               int32_t max_code_units) {
  int32_t length;
  int32_t position;
  int32_t stop;
  int32_t prepend;
  int32_t replace_from;
  udontokenizersStatus_t status;
  if (!model || !encoder || !workspace || !position_io || max_code_units <= 0)
    return UDONTOKENIZERS_STATUS_INVALID_ARGUMENT;
  if (encoder->status != UDONTOKENIZERS_STATUS_SUCCESS)
    return (udontokenizersStatus_t)encoder->status;
#if !(defined(__clang__) && defined(__SPIR__))
  if (use_units ? (!units && units_length) : !string)
    return UDONTOKENIZERS_STATUS_INVALID_ARGUMENT;
#endif
  if (encoder->phase != UDONTOKENIZERS_PHASE_TEXT ||
      (use_units && units_length < 0))
    return UDONTOKENIZERS_STATUS_INVALID_ARGUMENT;
  length = source_length(string, units, units_length, use_units);
  position = *position_io;
  if (position < 0 || position > length)
    return UDONTOKENIZERS_STATUS_INVALID_ARGUMENT;
  stop = length;
  if (max_code_units < length - position)
    stop = position + max_code_units;
  prepend = model->normalizer_prepend_scalar;
  replace_from = model->normalizer_replace_from_scalar;
  for (; position < stop; ++position) {
    uint32_t scalar = source_unit(string, units, use_units, position);
    int32_t added = match_added_token(model, string, units, use_units, length,
                                      position, scalar);
    if (added >= 0) {
      status =
          append_symbol(encoder, workspace, -1 - model->added_token_ids[added]);
      if (status != UDONTOKENIZERS_STATUS_SUCCESS)
        return fail(encoder, status);
      encoder->segment_started = 0;
      position += model->added_token_offsets[added + 1] -
                  model->added_token_offsets[added] - 1;
      continue;
    }
    if (scalar >= 0xd800 && scalar <= 0xdbff) {
      uint32_t low;
      if (position + 1 >= length)
        return fail(encoder, UDONTOKENIZERS_STATUS_INVALID_UTF16);
      low = source_unit(string, units, use_units, ++position);
      if (low < 0xdc00 || low > 0xdfff)
        return fail(encoder, UDONTOKENIZERS_STATUS_INVALID_UTF16);
      scalar = 0x10000 + ((scalar - 0xd800) << 10) + (low - 0xdc00);
    } else if (scalar >= 0xdc00 && scalar <= 0xdfff) {
      return fail(encoder, UDONTOKENIZERS_STATUS_INVALID_UTF16);
    }
    if (!encoder->segment_started) {
      encoder->segment_started = 1;
      if (prepend >= 0) {
        status = append_scalar(model, encoder, workspace, (uint32_t)prepend);
        if (status != UDONTOKENIZERS_STATUS_SUCCESS)
          return fail(encoder, status);
      }
    }
    if (replace_from >= 0 && scalar == (uint32_t)replace_from)
      scalar = (uint32_t)model->normalizer_replace_to_scalar;
    status = append_scalar(model, encoder, workspace, scalar);
    if (status != UDONTOKENIZERS_STATUS_SUCCESS)
      return fail(encoder, status);
  }
  *position_io = position;
  return UDONTOKENIZERS_STATUS_SUCCESS;
}

static udontokenizersStatus_t heap_push(udontokenizers_encoder_t *encoder,
                                        int32_t *heap, int32_t rank,
                                        int32_t position) {
  int32_t bits = encoder->position_bits;
  int32_t index = encoder->heap_count;
  int32_t key;
  if (rank < 0 || (rank >> (31 - bits)) != 0 || index >= 2 * encoder->capacity)
    return UDONTOKENIZERS_STATUS_INVALID_MODEL;
  key = (rank << bits) | position;
  encoder->heap_count = index + 1;
  while (index != 0) {
    int32_t parent = (index - 1) >> 1;
    int32_t parent_key = heap[parent];
    if (parent_key <= key)
      break;
    heap[index] = parent_key;
    index = parent;
  }
  heap[index] = key;
  return UDONTOKENIZERS_STATUS_SUCCESS;
}

static int32_t heap_pop(udontokenizers_encoder_t *encoder, int32_t *heap) {
  int32_t top = heap[0];
  int32_t count = encoder->heap_count - 1;
  int32_t last = heap[count];
  int32_t index = 0;
  encoder->heap_count = count;
  for (;;) {
    int32_t child = 2 * index + 1;
    int32_t child_key;
    if (child >= count)
      break;
    child_key = heap[child];
    if (child + 1 < count) {
      int32_t other_key = heap[child + 1];
      if (other_key < child_key) {
        child_key = other_key;
        ++child;
      }
    }
    if (child_key >= last)
      break;
    heap[index] = child_key;
    index = child;
  }
  if (count > 0)
    heap[index] = last;
  return top;
}

static udontokenizersStatus_t encoder_step(const udontokenizers_model_t *model,
                                           udontokenizers_encoder_t *encoder,
                                           int32_t *workspace, int32_t max_work,
                                           int32_t *finished) {
  const int32_t *lefts;
  const int32_t *rights;
  const int32_t *slots;
  const int32_t *results;
  const int32_t *ranks;
  int32_t *previous;
  int32_t *next;
  int32_t *candidates;
  int32_t *heap;
  int32_t merge_count;
  int32_t mask;
  int32_t multiplier;
  int32_t capacity;
  int32_t count;
  int32_t work = 0;
  udontokenizersStatus_t status;
  if (!model || !encoder || !workspace || !finished || max_work <= 0)
    return UDONTOKENIZERS_STATUS_INVALID_ARGUMENT;
  *finished = 0;
  if (encoder->status != UDONTOKENIZERS_STATUS_SUCCESS)
    return (udontokenizersStatus_t)encoder->status;
  if (encoder->phase == UDONTOKENIZERS_PHASE_FINISHED) {
    *finished = 1;
    return UDONTOKENIZERS_STATUS_SUCCESS;
  }
  lefts = model->merge_left_token_ids;
  rights = model->merge_right_token_ids;
  slots = model->merge_hash_slots;
  results = model->merge_result_token_ids;
  ranks = model->merge_ranks;
  merge_count = model->merge_count;
  mask = model->merge_hash_mask;
  multiplier = model->merge_hash_multiplier;
  capacity = encoder->capacity;
  count = encoder->count;
  previous = workspace + capacity;
  next = previous + capacity;
  candidates = next + capacity;
  heap = candidates + capacity;
  if (encoder->phase == UDONTOKENIZERS_PHASE_TEXT) {
    encoder->phase = UDONTOKENIZERS_PHASE_LINK;
    encoder->cursor = 0;
    encoder->heap_count = 0;
  }
  if (encoder->phase == UDONTOKENIZERS_PHASE_LINK) {
    int32_t cursor = encoder->cursor;
    for (; cursor < count; ++cursor) {
      int32_t candidate = -1;
      if (work >= max_work) {
        encoder->cursor = cursor;
        return UDONTOKENIZERS_STATUS_SUCCESS;
      }
      ++work;
      previous[cursor] = cursor - 1;
      if (cursor + 1 < count) {
        int32_t left = workspace[cursor];
        int32_t right = workspace[cursor + 1];
        next[cursor] = cursor + 1;
        if (left >= 0 && right >= 0)
          candidate = find_merge(lefts, rights, merge_count, slots, mask,
                                 multiplier, left, right);
      } else {
        next[cursor] = -1;
      }
      candidates[cursor] = candidate;
      if (candidate >= 0) {
        status = heap_push(encoder, heap, ranks[candidate], cursor);
        if (status != UDONTOKENIZERS_STATUS_SUCCESS)
          return fail(encoder, status);
      }
    }
    encoder->phase = UDONTOKENIZERS_PHASE_MERGE;
  }
  if (encoder->phase == UDONTOKENIZERS_PHASE_MERGE) {
    int32_t bits = encoder->position_bits;
    int32_t position_mask = encoder->position_mask;
    while (encoder->heap_count > 0) {
      int32_t key;
      int32_t position;
      int32_t candidate;
      int32_t right;
      int32_t after;
      int32_t before;
      int32_t merged;
      if (work >= max_work)
        return UDONTOKENIZERS_STATUS_SUCCESS;
      ++work;
      key = heap_pop(encoder, heap);
      position = key & position_mask;
      candidate = candidates[position];
      if (candidate < 0 || ranks[candidate] != (key >> bits))
        continue;
      merged = results[candidate];
      right = next[position];
      after = next[right];
      workspace[position] = merged;
      candidates[right] = -1;
      next[position] = after;
      candidate = -1;
      if (after >= 0) {
        int32_t following = workspace[after];
        previous[after] = position;
        if (following >= 0)
          candidate = find_merge(lefts, rights, merge_count, slots, mask,
                                 multiplier, merged, following);
      }
      candidates[position] = candidate;
      if (candidate >= 0) {
        status = heap_push(encoder, heap, ranks[candidate], position);
        if (status != UDONTOKENIZERS_STATUS_SUCCESS)
          return fail(encoder, status);
      }
      before = previous[position];
      if (before >= 0) {
        int32_t preceding = workspace[before];
        if (preceding >= 0) {
          candidate = find_merge(lefts, rights, merge_count, slots, mask,
                                 multiplier, preceding, merged);
          candidates[before] = candidate;
          if (candidate >= 0) {
            status = heap_push(encoder, heap, ranks[candidate], before);
            if (status != UDONTOKENIZERS_STATUS_SUCCESS)
              return fail(encoder, status);
          }
        }
      }
    }
    encoder->phase = UDONTOKENIZERS_PHASE_EMIT;
    encoder->cursor = 0;
    /* The emptied heap count becomes the number of emitted tokens. */
    encoder->heap_count = 0;
  }
  {
    /* Compact the live symbols to the front. A token is never written past
     * the symbol it came from, so the walk reads only unwritten symbols. */
    int32_t cursor = encoder->cursor;
    int32_t emitted = encoder->heap_count;
    while (cursor >= 0 && cursor < count) {
      int32_t symbol;
      if (work >= max_work) {
        encoder->cursor = cursor;
        encoder->heap_count = emitted;
        return UDONTOKENIZERS_STATUS_SUCCESS;
      }
      ++work;
      symbol = workspace[cursor];
      workspace[emitted++] = symbol < 0 ? -1 - symbol : symbol;
      cursor = next[cursor];
    }
    encoder->cursor = 0;
    encoder->heap_count = 0;
    encoder->count = emitted;
  }
  encoder->phase = UDONTOKENIZERS_PHASE_FINISHED;
  *finished = 1;
  return UDONTOKENIZERS_STATUS_SUCCESS;
}

udontokenizersStatus_t udontokenizersEncoderBeginRaw(
    const udontokenizers_model_t *model, udontokenizers_encoder_t *encoder,
    int32_t *workspace, int32_t workspace_words, int32_t add_special_tokens) {
  int32_t position_bits = 31;
  int32_t position_mask = 0x7fffffff;
  int32_t ranks;
  int32_t capacity;
  if (!model || !encoder || !workspace || workspace_words < 0)
    return UDONTOKENIZERS_STATUS_INVALID_ARGUMENT;
  encoder->count = 0;
  encoder->capacity = 0;
  encoder->segment_started = 0;
  encoder->phase = UDONTOKENIZERS_PHASE_TEXT;
  encoder->cursor = 0;
  encoder->heap_count = 0;
  encoder->position_bits = 31;
  encoder->position_mask = 0x7fffffff;
  encoder->status = UDONTOKENIZERS_STATUS_SUCCESS;
  if (!valid_model(model))
    return fail(encoder, UDONTOKENIZERS_STATUS_INVALID_MODEL);
  /* Ranks are below merge_count. Every bit they need is taken from the
   * position field of a non-negative heap key. */
  for (ranks = model->merge_count; ranks > 0; ranks >>= 1) {
    --position_bits;
    position_mask >>= 1;
  }
  capacity = workspace_words / 6;
  if (capacity > position_mask)
    capacity = position_mask;
  encoder->capacity = capacity;
  encoder->position_bits = position_bits;
  encoder->position_mask = position_mask;
  if (add_special_tokens && model->bos_token_id >= 0) {
    udontokenizersStatus_t status =
        append_symbol(encoder, workspace, -1 - model->bos_token_id);
    if (status != UDONTOKENIZERS_STATUS_SUCCESS)
      return fail(encoder, status);
  }
  return UDONTOKENIZERS_STATUS_SUCCESS;
}

udontokenizersStatus_t
udontokenizersEncoderAppendRaw(const udontokenizers_model_t *model,
                               udontokenizers_encoder_t *encoder,
                               udon_string_t text, int32_t *workspace) {
  int32_t position = 0;
  return encoder_append(model, encoder, text, 0, 0, 0, workspace, &position,
                        UDONTOKENIZERS_UNBOUNDED);
}

udontokenizersStatus_t udontokenizersEncoderAppendPartRaw(
    const udontokenizers_model_t *model, udontokenizers_encoder_t *encoder,
    udon_string_t text, int32_t *position, int32_t max_code_units,
    int32_t *finished, int32_t *workspace) {
  udontokenizersStatus_t status;
  if (!finished)
    return UDONTOKENIZERS_STATUS_INVALID_ARGUMENT;
  *finished = 0;
  status = encoder_append(model, encoder, text, 0, 0, 0, workspace, position,
                          max_code_units);
  if (status == UDONTOKENIZERS_STATUS_SUCCESS)
    *finished = *position >= udonStringLength(text);
  return status;
}

udontokenizersStatus_t udontokenizersEncoderAppendCodeUnitsRaw(
    const udontokenizers_model_t *model, udontokenizers_encoder_t *encoder,
    const uint16_t *code_units, int32_t length, int32_t *workspace) {
  int32_t position = 0;
  return encoder_append(model, encoder, (udon_string_t)0, code_units, length, 1,
                        workspace, &position, UDONTOKENIZERS_UNBOUNDED);
}

udontokenizersStatus_t udontokenizersEncoderStepRaw(
    const udontokenizers_model_t *model, udontokenizers_encoder_t *encoder,
    int32_t *workspace, int32_t max_work, int32_t *finished) {
  return encoder_step(model, encoder, workspace, max_work, finished);
}

udontokenizersStatus_t
udontokenizersEncoderFinishRaw(const udontokenizers_model_t *model,
                               udontokenizers_encoder_t *encoder,
                               int32_t *workspace, int32_t *output_count) {
  int32_t finished = 0;
  udontokenizersStatus_t status = UDONTOKENIZERS_STATUS_SUCCESS;
  if (!output_count)
    return UDONTOKENIZERS_STATUS_INVALID_ARGUMENT;
  while (status == UDONTOKENIZERS_STATUS_SUCCESS && !finished)
    status = encoder_step(model, encoder, workspace, UDONTOKENIZERS_UNBOUNDED,
                          &finished);
  if (status == UDONTOKENIZERS_STATUS_SUCCESS)
    *output_count = encoder->count;
  return status;
}

static udontokenizersStatus_t
encode(const udontokenizers_model_t *model, udon_string_t input,
       udon_int32_array_t managed_output, int32_t *raw_output, int32_t use_raw,
       int32_t output_capacity, int32_t *output_count, int32_t *workspace,
       int32_t workspace_words) {
  udontokenizers_encoder_t encoder;
  udontokenizersStatus_t status;
  int32_t count = 0;
  int32_t index;
  if (!model || !output_count || output_capacity < 0)
    return UDONTOKENIZERS_STATUS_INVALID_ARGUMENT;
#if !(defined(__clang__) && defined(__SPIR__))
  if (!input || (use_raw ? !raw_output : !managed_output))
    return UDONTOKENIZERS_STATUS_INVALID_ARGUMENT;
#endif
  status = udontokenizersEncoderBeginRaw(model, &encoder, workspace,
                                         workspace_words, 1);
  if (status == UDONTOKENIZERS_STATUS_SUCCESS)
    status = udontokenizersEncoderAppendRaw(model, &encoder, input, workspace);
  if (status == UDONTOKENIZERS_STATUS_SUCCESS)
    status = udontokenizersEncoderFinishRaw(model, &encoder, workspace, &count);
  if (status != UDONTOKENIZERS_STATUS_SUCCESS)
    return status;
  if (count > output_capacity)
    return UDONTOKENIZERS_STATUS_OUTPUT_TOO_SMALL;
  if (use_raw) {
    for (index = 0; index < count; ++index)
      raw_output[index] = workspace[index];
  } else {
    for (index = 0; index < count; ++index)
      udonInt32ArraySet(managed_output, index, workspace[index]);
  }
  *output_count = count;
  return UDONTOKENIZERS_STATUS_SUCCESS;
}

udontokenizersStatus_t
udontokenizersEncode(const udontokenizers_model_t *model, udon_string_t input,
                     udon_int32_array_t output, int32_t output_capacity,
                     int32_t *output_count, int32_t *workspace,
                     int32_t workspace_words) {
  if (output_capacity < 0 || output_capacity > udonInt32ArrayLength(output))
    return UDONTOKENIZERS_STATUS_INVALID_ARGUMENT;
  return encode(model, input, output, 0, 0, output_capacity, output_count,
                workspace, workspace_words);
}

udontokenizersStatus_t
udontokenizersEncodeRaw(const udontokenizers_model_t *model,
                        udon_string_t input, int32_t *output,
                        int32_t output_capacity, int32_t *output_count,
                        int32_t *workspace, int32_t workspace_words) {
  return encode(model, input, (udon_int32_array_t)0, output, 1, output_capacity,
                output_count, workspace, workspace_words);
}

static int32_t valid_chat_role(const udontokenizers_chat_template_t *chat,
                               int32_t role) {
  if (!chat || !chat->code_units || !chat->role_prefix_offsets ||
      !chat->role_prefix_lengths || !chat->role_suffix_offsets ||
      !chat->role_suffix_lengths || role < 0 ||
      role >= UDONTOKENIZERS_CHAT_ROLE_COUNT)
    return 0;
  return chat->role_prefix_lengths[role] >= 0 &&
         chat->role_suffix_lengths[role] >= 0;
}

udontokenizersStatus_t udontokenizersEncoderAppendChatMessagePartRaw(
    const udontokenizers_model_t *model,
    const udontokenizers_chat_template_t *chat,
    udontokenizers_encoder_t *encoder, int32_t role, udon_string_t content,
    int32_t *position, int32_t max_code_units, int32_t *finished,
    int32_t *workspace) {
  int32_t prefix_length;
  int32_t content_length;
  int32_t suffix_length;
  int32_t start;
  int32_t local;
  udontokenizersStatus_t status = UDONTOKENIZERS_STATUS_SUCCESS;
  if (!position || !finished || max_code_units <= 0 ||
      !valid_chat_role(chat, role))
    return UDONTOKENIZERS_STATUS_INVALID_ARGUMENT;
  *finished = 0;
  prefix_length = chat->role_prefix_lengths[role];
  suffix_length = chat->role_suffix_lengths[role];
  content_length = udonStringLength(content);
  start = *position;
  if (start < 0 || start > prefix_length + content_length + suffix_length)
    return UDONTOKENIZERS_STATUS_INVALID_ARGUMENT;
  if (*position < prefix_length) {
    local = *position;
    status = encoder_append(model, encoder, (udon_string_t)0,
                            chat->code_units + chat->role_prefix_offsets[role],
                            prefix_length, 1, workspace, &local,
                            max_code_units - (*position - start));
    *position = local;
  }
  if (status == UDONTOKENIZERS_STATUS_SUCCESS && *position >= prefix_length &&
      *position < prefix_length + content_length &&
      *position - start < max_code_units) {
    local = *position - prefix_length;
    status = encoder_append(model, encoder, content, 0, 0, 0, workspace, &local,
                            max_code_units - (*position - start));
    *position = prefix_length + local;
  }
  if (status == UDONTOKENIZERS_STATUS_SUCCESS &&
      *position >= prefix_length + content_length &&
      *position < prefix_length + content_length + suffix_length &&
      *position - start < max_code_units) {
    local = *position - prefix_length - content_length;
    status = encoder_append(model, encoder, (udon_string_t)0,
                            chat->code_units + chat->role_suffix_offsets[role],
                            suffix_length, 1, workspace, &local,
                            max_code_units - (*position - start));
    *position = prefix_length + content_length + local;
  }
  if (status == UDONTOKENIZERS_STATUS_SUCCESS)
    *finished = *position >= prefix_length + content_length + suffix_length;
  return status;
}

udontokenizersStatus_t udontokenizersEncoderAppendChatMessageRaw(
    const udontokenizers_model_t *model,
    const udontokenizers_chat_template_t *chat,
    udontokenizers_encoder_t *encoder, int32_t role, udon_string_t content,
    int32_t *workspace) {
  int32_t position = 0;
  int32_t finished = 0;
  return udontokenizersEncoderAppendChatMessagePartRaw(
      model, chat, encoder, role, content, &position, UDONTOKENIZERS_UNBOUNDED,
      &finished, workspace);
}

/* Length of the end-of-sequence added token's text when `role`'s suffix
 * starts with it, else -1. */
static int32_t
chat_suffix_eos_length(const udontokenizers_model_t *model,
                       const udontokenizers_chat_template_t *chat,
                       int32_t role) {
  const int32_t *offsets = model->added_token_offsets;
  int32_t suffix_offset = chat->role_suffix_offsets[role];
  int32_t token;
  if (model->eos_token_id < 0)
    return -1;
  for (token = 0; token < model->added_token_count; ++token) {
    int32_t begin = offsets[token];
    int32_t length = offsets[token + 1] - begin;
    int32_t index;
    if (model->added_token_ids[token] != model->eos_token_id || length <= 0 ||
        length > chat->role_suffix_lengths[role])
      continue;
    for (index = 0; index < length; ++index)
      if (chat->code_units[suffix_offset + index] !=
          model->added_token_code_units[begin + index])
        break;
    if (index == length)
      return length;
  }
  return -1;
}

udontokenizersStatus_t udontokenizersEncoderAppendChatTurnPartRaw(
    const udontokenizers_model_t *model,
    const udontokenizers_chat_template_t *chat,
    udontokenizers_encoder_t *encoder, int32_t previous_role, int32_t role,
    udon_string_t content, int32_t *position, int32_t max_code_units,
    int32_t *finished, int32_t *workspace) {
  int32_t eos_length;
  int32_t rest_length;
  int32_t start;
  int32_t local;
  udontokenizersStatus_t status = UDONTOKENIZERS_STATUS_SUCCESS;
  if (!model || !encoder || !position || !finished || max_code_units <= 0 ||
      !valid_chat_role(chat, previous_role) || !valid_chat_role(chat, role))
    return UDONTOKENIZERS_STATUS_INVALID_ARGUMENT;
  *finished = 0;
  eos_length = chat_suffix_eos_length(model, chat, previous_role);
  if (eos_length < 0)
    return UDONTOKENIZERS_STATUS_INVALID_ARGUMENT;
  rest_length = chat->role_suffix_lengths[previous_role] - eos_length;
  start = *position;
  if (start < 0)
    return UDONTOKENIZERS_STATUS_INVALID_ARGUMENT;
  /* The end-of-sequence added token already ended a segment. */
  if (!start && encoder->phase == UDONTOKENIZERS_PHASE_TEXT)
    encoder->segment_started = 0;
  if (start < rest_length) {
    local = start;
    status = encoder_append(model, encoder, (udon_string_t)0,
                            chat->code_units +
                                chat->role_suffix_offsets[previous_role] +
                                eos_length,
                            rest_length, 1, workspace, &local, max_code_units);
    *position = local;
  }
  if (status == UDONTOKENIZERS_STATUS_SUCCESS && *position >= rest_length &&
      *position - start < max_code_units) {
    local = *position - rest_length;
    status = udontokenizersEncoderAppendChatMessagePartRaw(
        model, chat, encoder, role, content, &local,
        max_code_units - (*position - start), finished, workspace);
    *position = rest_length + local;
  }
  return status;
}

udontokenizersStatus_t udontokenizersEncoderAppendChatTurnRaw(
    const udontokenizers_model_t *model,
    const udontokenizers_chat_template_t *chat,
    udontokenizers_encoder_t *encoder, int32_t previous_role, int32_t role,
    udon_string_t content, int32_t *workspace) {
  int32_t position = 0;
  int32_t finished = 0;
  return udontokenizersEncoderAppendChatTurnPartRaw(
      model, chat, encoder, previous_role, role, content, &position,
      UDONTOKENIZERS_UNBOUNDED, &finished, workspace);
}

udontokenizersStatus_t udontokenizersEncoderAppendChatGenerationPromptRaw(
    const udontokenizers_model_t *model,
    const udontokenizers_chat_template_t *chat,
    udontokenizers_encoder_t *encoder, int32_t *workspace) {
  int32_t position = 0;
  if (!chat || !chat->code_units || chat->generation_prompt_length < 0)
    return UDONTOKENIZERS_STATUS_INVALID_ARGUMENT;
  if (!chat->generation_prompt_length)
    return UDONTOKENIZERS_STATUS_SUCCESS;
  return encoder_append(model, encoder, (udon_string_t)0,
                        chat->code_units + chat->generation_prompt_offset,
                        chat->generation_prompt_length, 1, workspace, &position,
                        UDONTOKENIZERS_UNBOUNDED);
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

/* Makes every written unit final. Called only while no byte-fallback run is
 * open. The leading strip is decided here, on units that can no longer
 * change: while it is still pending no unit is final yet, so removing a
 * leading unit moves only the units of this call. */
static void finalize_units(const udontokenizers_model_t *model,
                           udontokenizers_decoder_t *decoder, int32_t *output) {
  while (decoder->strip_pending > 0 && decoder->count > 0) {
    int32_t index;
    if (output[0] != model->decoder_strip_scalar) {
      decoder->strip_pending = 0;
      break;
    }
    for (index = 1; index < decoder->count; ++index)
      output[index - 1] = output[index];
    --decoder->count;
    --decoder->strip_pending;
  }
  decoder->final_count = decoder->count;
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
  decoder->final_count = 0;
  decoder->strip_pending = 0;
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
  if (model->decoder_strip_scalar >= 0)
    decoder->strip_pending = model->decoder_strip_count;
  return UDONTOKENIZERS_STATUS_SUCCESS;
}

udontokenizersStatus_t
udontokenizersDecoderAppendRaw(const udontokenizers_model_t *model,
                               udontokenizers_decoder_t *decoder,
                               int32_t token_id, int32_t *output) {
  int32_t kind;
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
    if (status == UDONTOKENIZERS_STATUS_SUCCESS) {
      const uint16_t *code_units = model->piece_code_units;
      int32_t index = model->piece_offsets[token_id];
      int32_t end = model->piece_offsets[token_id + 1];
      int32_t count = decoder->count;
      if (end - index > decoder->capacity - count) {
        status = UDONTOKENIZERS_STATUS_OUTPUT_TOO_SMALL;
      } else {
        for (; index < end; ++index)
          output[count++] = code_units[index];
        decoder->count = count;
        finalize_units(model, decoder, output);
      }
    }
  }
  if (status != UDONTOKENIZERS_STATUS_SUCCESS)
    decoder->status = status;
  return status;
}

udontokenizersStatus_t
udontokenizersDecoderStepRaw(const udontokenizers_model_t *model,
                             udontokenizers_decoder_t *decoder,
                             int32_t token_id, int32_t *output,
                             int32_t *chunk_offset, int32_t *chunk_length) {
  int32_t before;
  udontokenizersStatus_t status;
  if (!decoder || !chunk_offset || !chunk_length)
    return UDONTOKENIZERS_STATUS_INVALID_ARGUMENT;
  before = decoder->final_count;
  *chunk_offset = before;
  *chunk_length = 0;
  status = udontokenizersDecoderAppendRaw(model, decoder, token_id, output);
  if (status == UDONTOKENIZERS_STATUS_SUCCESS)
    *chunk_length = decoder->final_count - before;
  return status;
}

udontokenizersStatus_t
udontokenizersDecoderFinishRaw(const udontokenizers_model_t *model,
                               udontokenizers_decoder_t *decoder,
                               int32_t *output, int32_t *output_count) {
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
  finalize_units(model, decoder, output);
  *output_count = decoder->count;
  return UDONTOKENIZERS_STATUS_SUCCESS;
}
