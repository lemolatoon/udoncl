#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

size_t udondnn_large_pointwise_cpu_gid_x;
size_t udondnn_large_pointwise_cpu_gid_y;

#include "../../libraries/udondnn/device/large_pointwise.cl"

int main() {
  constexpr int words = 16 * 24 * 24;
  constexpr int arenaWords = 16384;
  std::vector<float> input(arenaWords, 0.0f);
  std::vector<float> bias(arenaWords, 0.0f);
  std::vector<float> previous(arenaWords, 0.0f);
  std::vector<float> combined(arenaWords, -1.0f);
  std::vector<float> activated(arenaWords, -1.0f);
  for (int index = 0; index < words; ++index) {
    input[index] = static_cast<float>((index % 31) - 15) / 7.0f;
    previous[index] = static_cast<float>((index % 9) - 4) / 13.0f;
  }
  for (int channel = 0; channel < 16; ++channel)
    bias[channel] = static_cast<float>(channel - 8) / 5.0f;

  for (udondnn_large_pointwise_cpu_gid_y = 0;
       udondnn_large_pointwise_cpu_gid_y < 2;
       ++udondnn_large_pointwise_cpu_gid_y)
    for (udondnn_large_pointwise_cpu_gid_x = 0;
         udondnn_large_pointwise_cpu_gid_x < 8192;
         ++udondnn_large_pointwise_cpu_gid_x)
      udondnn_large_op_tensor(
          input.data(), bias.data(), previous.data(), combined.data(), words,
          1 | (16 << 16), 24 | (24 << 16), 1 | (16 << 16), 1 | (1 << 16),
          1 | (16 << 16), 24 | (24 << 16), 0, 1.25f, -0.5f, 0.25f);

  float maximumError = 0.0f;
  for (int index = 0; index < words; ++index) {
    int channel = (index / (24 * 24)) % 16;
    float expected =
        1.25f * input[index] - 0.5f * bias[channel] + 0.25f * previous[index];
    maximumError = std::max(maximumError, std::abs(combined[index] - expected));
  }

  for (udondnn_large_pointwise_cpu_gid_y = 0;
       udondnn_large_pointwise_cpu_gid_y < 2;
       ++udondnn_large_pointwise_cpu_gid_y)
    for (udondnn_large_pointwise_cpu_gid_x = 0;
         udondnn_large_pointwise_cpu_gid_x < 8192;
         ++udondnn_large_pointwise_cpu_gid_x)
      udondnn_large_activation_forward(combined.data(), previous.data(),
                                       activated.data(), words, 1, 0, 0.0f,
                                       0.75f, -0.25f);
  for (int index = 0; index < words; ++index) {
    float expected =
        0.75f * std::max(combined[index], 0.0f) - 0.25f * previous[index];
    maximumError =
        std::max(maximumError, std::abs(activated[index] - expected));
  }
  for (int index = words; index < arenaWords; ++index)
    maximumError = std::max(maximumError, std::max(std::abs(combined[index]),
                                                   std::abs(activated[index])));

  constexpr int attentionWords = 1 * 32 * 64 * 64;
  std::vector<float> attention(attentionWords);
  std::vector<float> attentionOutput(attentionWords, -1.0f);
  std::vector<float> unusedPrevious(1, 0.0f);
  std::vector<float> scale(1, 0.125f);
  for (int index = 0; index < attentionWords; ++index)
    attention[index] = static_cast<float>((index % 41) - 20) / 13.0f;
  for (udondnn_large_pointwise_cpu_gid_y = 0;
       udondnn_large_pointwise_cpu_gid_y < 16;
       ++udondnn_large_pointwise_cpu_gid_y)
    for (udondnn_large_pointwise_cpu_gid_x = 0;
         udondnn_large_pointwise_cpu_gid_x < 8192;
         ++udondnn_large_pointwise_cpu_gid_x)
      udondnn_large_op_tensor(attention.data(), scale.data(),
                              unusedPrevious.data(), attentionOutput.data(),
                              attentionWords, 1 | (32 << 16), 64 | (64 << 16),
                              1 | (1 << 16), 1 | (1 << 16), 1 | (32 << 16),
                              64 | (64 << 16), 1, 1.0f, 1.0f, 0.0f);
  for (int index = 0; index < attentionWords; ++index)
    maximumError =
        std::max(maximumError, std::abs(attentionOutput[index] -
                                        attention[index] * scale[0]));
  std::cout << "large_pointwise_outputs=" << words << '\n';
  std::cout << "large_pointwise_attention_outputs=" << attentionWords << '\n';
  std::cout << "large_pointwise_max_error=" << maximumError << '\n';
  return maximumError < 0.00001f ? 0 : 1;
}
