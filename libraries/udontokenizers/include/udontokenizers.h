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
  UDONTOKENIZERS_STATUS_OUTPUT_TOO_SMALL = 4
} udontokenizersStatus_t;

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
} udontokenizers_model_t;

udontokenizersStatus_t udontokenizersEncode(const udontokenizers_model_t *model,
                                            udon_string_t input,
                                            udon_int32_array_t output,
                                            int32_t output_capacity,
                                            int32_t *output_count);

udontokenizersStatus_t
udontokenizersEncodeRaw(const udontokenizers_model_t *model,
                        udon_string_t input, int32_t *output,
                        int32_t output_capacity, int32_t *output_count);

#ifdef __cplusplus
}
#endif

#endif
