#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
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

  // The ONNX planner folds a Mul read only by one Add into that Add: one
  // dispatch computes product + previous. It must equal the Mul dispatch
  // followed by the Add dispatch bit for bit, with either Add operand order.
  constexpr int rotaryWords = 1 * 32 * 1 * 64;
  std::vector<float> rotaryLeft(arenaWords, 0.0f);
  std::vector<float> rotaryRight(arenaWords, 0.0f);
  std::vector<float> rotaryCos(64);
  std::vector<float> rotarySin(64);
  std::vector<float> leftProduct(arenaWords, -1.0f);
  std::vector<float> rightProduct(arenaWords, -1.0f);
  std::vector<float> unfolded(arenaWords, -1.0f);
  std::vector<float> unfoldedSwapped(arenaWords, -1.0f);
  std::vector<float> folded(arenaWords, -1.0f);
  for (int index = 0; index < rotaryWords; ++index) {
    rotaryLeft[index] = std::sin(0.37f * static_cast<float>(index)) * 3.25f;
    rotaryRight[index] = std::cos(0.11f * static_cast<float>(index)) / 7.0f;
  }
  rotaryLeft[5] = 1.0e-30f;
  rotaryRight[5] = 3.0e38f;
  rotaryLeft[6] = -0.0f;
  for (int index = 0; index < 64; ++index) {
    rotaryCos[index] = std::cos(1.7f * static_cast<float>(index));
    rotarySin[index] = std::sin(1.7f * static_cast<float>(index));
  }
  const int rotaryNc = 1 | (32 << 16);
  const int rotaryHw = 1 | (64 << 16);
  const int rowNc = 1 | (1 << 16);
  auto opTensor = [&](const std::vector<float> &a, int aNc,
                      const std::vector<float> &b, int bNc,
                      const std::vector<float> &prior,
                      std::vector<float> &result, int operation, float beta) {
    for (udondnn_large_pointwise_cpu_gid_y = 0;
         udondnn_large_pointwise_cpu_gid_y < 2;
         ++udondnn_large_pointwise_cpu_gid_y)
      for (udondnn_large_pointwise_cpu_gid_x = 0;
           udondnn_large_pointwise_cpu_gid_x < 8192;
           ++udondnn_large_pointwise_cpu_gid_x)
        udondnn_large_op_tensor(a.data(), b.data(), prior.data(), result.data(),
                                rotaryWords, aNc, rotaryHw, bNc, rotaryHw,
                                rotaryNc, rotaryHw, operation | (1 << 8), 1.0f,
                                1.0f, beta);
  };
  opTensor(rotaryLeft, rotaryNc, rotaryCos, rowNc, rotaryLeft, leftProduct, 1,
           0.0f);
  opTensor(rotaryRight, rotaryNc, rotarySin, rowNc, rotaryRight, rightProduct,
           1, 0.0f);
  opTensor(leftProduct, rotaryNc, rightProduct, rotaryNc, leftProduct, unfolded,
           0, 0.0f);
  opTensor(rightProduct, rotaryNc, leftProduct, rotaryNc, rightProduct,
           unfoldedSwapped, 0, 0.0f);
  opTensor(rotaryRight, rotaryNc, rotarySin, rowNc, leftProduct, folded, 1,
           1.0f);
  bool foldedIdentical = std::memcmp(folded.data(), unfolded.data(),
                                     arenaWords * sizeof(float)) == 0 &&
                         std::memcmp(folded.data(), unfoldedSwapped.data(),
                                     arenaWords * sizeof(float)) == 0;
  std::cout << "large_pointwise_folded_mul_add_identical=" << foldedIdentical
            << '\n';
  if (!foldedIdentical)
    return 1;
  std::cout << "large_pointwise_outputs=" << words << '\n';
  std::cout << "large_pointwise_attention_outputs=" << attentionWords << '\n';
  std::cout << "large_pointwise_max_error=" << maximumError << '\n';
  return maximumError < 0.00001f ? 0 : 1;
}
