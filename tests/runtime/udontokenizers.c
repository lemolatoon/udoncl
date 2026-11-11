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

int main(void) {
  static const uint32_t scalars[] = {0x65e5, 0x672c, 0x8a9e, 0x20};
  static const int32_t scalar_tokens[] = {10, 11, 12, 13};
  static const udontokenizers_merge_t merges[] = {
      {10, 11, 20, 0},
      {20, 12, 30, 1},
  };
  int32_t byte_tokens[256];
  udontokenizers_model_t model;
  const uint16_t text_data[] = {0x65e5, 0x672c, 0x8a9e, 0x20, 0xd83c, 0xdf5c};
  udon_string_native_t text = {text_data, 6};
  int32_t token_data[8] = {0};
  udon_int32_array_native_t tokens = {token_data, 8};
  int32_t token_count = -1;
  int32_t index;

  for (index = 0; index < 256; ++index)
    byte_tokens[index] = 1000 + index;
  model.scalar_values = scalars;
  model.scalar_token_ids = scalar_tokens;
  model.scalar_count = 4;
  model.merges = merges;
  model.merge_count = 2;
  model.byte_token_ids = byte_tokens;

  EXPECT(udontokenizersEncode(&model, &text, &tokens, 8, &token_count) ==
         UDONTOKENIZERS_STATUS_SUCCESS);
  EXPECT(token_count == 6);
  EXPECT(token_data[0] == 30);
  EXPECT(token_data[1] == 13);
  EXPECT(token_data[2] == 1240);
  EXPECT(token_data[3] == 1159);
  EXPECT(token_data[4] == 1141);
  EXPECT(token_data[5] == 1156);

  EXPECT(udontokenizersEncode(&model, &text, &tokens, 5, &token_count) ==
         UDONTOKENIZERS_STATUS_OUTPUT_TOO_SMALL);
  return 0;
}
