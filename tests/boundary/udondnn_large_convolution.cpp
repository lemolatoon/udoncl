#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

size_t udondnn_large_convolution_cpu_gid_x;
size_t udondnn_large_convolution_cpu_gid_y;

#include "../../libraries/udondnn/device/large_convolution.cl"

int main() {
  constexpr int inputWords = 28 * 28;
  constexpr int filterWords = 16 * 5 * 5;
  constexpr int outputWords = 16 * 24 * 24;
  constexpr int arenaWords = 16384;
  std::vector<float> input(arenaWords, 0.0f);
  std::vector<float> filter(arenaWords, 0.0f);
  std::vector<float> previous(arenaWords, 0.0f);
  std::vector<float> output(arenaWords, -1.0f);
  for (int index = 0; index < inputWords; ++index)
    input[index] = static_cast<float>((index % 37) - 18) / 11.0f;
  for (int index = 0; index < filterWords; ++index)
    filter[index] = static_cast<float>((index % 17) - 8) / 19.0f;
  for (int index = 0; index < outputWords; ++index)
    previous[index] = static_cast<float>((index % 13) - 6) / 23.0f;

  for (udondnn_large_convolution_cpu_gid_y = 0;
       udondnn_large_convolution_cpu_gid_y < 2;
       ++udondnn_large_convolution_cpu_gid_y)
    for (udondnn_large_convolution_cpu_gid_x = 0;
         udondnn_large_convolution_cpu_gid_x < 8192;
         ++udondnn_large_convolution_cpu_gid_x)
      udondnn_large_convolution_forward(
          input.data(), filter.data(), previous.data(), output.data(),
          outputWords, 1 | (28 << 16), 28, 16 | (24 << 16), 24, 5 | (5 << 16),
          0, 1 | (1 << 16), 1 | (1 << 16), 1, 0.75f, -0.25f);

  float maximumError = 0.0f;
  for (int channel = 0; channel < 16; ++channel)
    for (int y = 0; y < 24; ++y)
      for (int x = 0; x < 24; ++x) {
        float sum = 0.0f;
        for (int filterY = 0; filterY < 5; ++filterY)
          for (int filterX = 0; filterX < 5; ++filterX)
            sum += input[(y + filterY) * 28 + x + filterX] *
                   filter[(channel * 5 + filterY) * 5 + filterX];
        int index = (channel * 24 + y) * 24 + x;
        float expected = 0.75f * sum - 0.25f * previous[index];
        maximumError =
            std::max(maximumError, std::abs(output[index] - expected));
      }
  for (int index = outputWords; index < arenaWords; ++index)
    maximumError = std::max(maximumError, std::abs(output[index]));
  std::cout << "large_convolution_outputs=" << outputWords << '\n';
  std::cout << "large_convolution_max_error=" << maximumError << '\n';
  return maximumError < 0.00001f ? 0 : 1;
}
