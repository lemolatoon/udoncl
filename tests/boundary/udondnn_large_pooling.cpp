#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

size_t udondnn_large_pooling_cpu_gid_x;
size_t udondnn_large_pooling_cpu_gid_y;

#include "../../libraries/udondnn/device/large_pooling.cl"

namespace {
constexpr int kWidth = 8192;
constexpr int kArenaWords = 16384;

float referenceAverage(const std::vector<float> &input, int channel,
                       int outputY, int outputX) {
  float sum = 0.0f;
  for (int kernelY = 0; kernelY < 8; ++kernelY)
    for (int kernelX = 0; kernelX < 8; ++kernelX) {
      int inputY = outputY * 8 + kernelY;
      int inputX = outputX * 8 + kernelX;
      size_t index = (static_cast<size_t>(channel) * 24 + inputY) * 24 + inputX;
      sum += input[index];
    }
  return sum / 64.0f;
}
} // namespace

int main() {
  constexpr int inputWords = 16 * 24 * 24;
  constexpr int outputWords = 16 * 3 * 3;
  std::vector<float> input(kArenaWords, 0.0f);
  std::vector<float> previous(kArenaWords, 0.0f);
  std::vector<float> output(kArenaWords, -1.0f);
  for (int index = 0; index < inputWords; ++index)
    input[index] = static_cast<float>((index % 97) - 48) / 13.0f;
  for (int index = 0; index < outputWords; ++index)
    previous[index] = static_cast<float>((index % 11) - 5) / 7.0f;

  for (udondnn_large_pooling_cpu_gid_y = 0; udondnn_large_pooling_cpu_gid_y < 2;
       ++udondnn_large_pooling_cpu_gid_y)
    for (udondnn_large_pooling_cpu_gid_x = 0;
         udondnn_large_pooling_cpu_gid_x < kWidth;
         ++udondnn_large_pooling_cpu_gid_x)
      udondnn_large_pooling_forward(
          input.data(), previous.data(), output.data(), outputWords, 16, 24, 24,
          3, 3, 8 | (8 << 16), 0, 8 | (8 << 16), 2, 1.25f, -0.5f);

  float maxError = 0.0f;
  for (int channel = 0; channel < 16; ++channel)
    for (int outputY = 0; outputY < 3; ++outputY)
      for (int outputX = 0; outputX < 3; ++outputX) {
        int index = (channel * 3 + outputY) * 3 + outputX;
        float expected =
            1.25f * referenceAverage(input, channel, outputY, outputX) -
            0.5f * previous[index];
        maxError = std::max(maxError, std::abs(output[index] - expected));
      }
  for (int index = outputWords; index < kArenaWords; ++index)
    maxError = std::max(maxError, std::abs(output[index]));

  std::cout << "large_pooling_outputs=" << outputWords << '\n';
  std::cout << "large_pooling_max_error=" << maxError << '\n';
  return maxError < 0.00001f ? 0 : 1;
}
