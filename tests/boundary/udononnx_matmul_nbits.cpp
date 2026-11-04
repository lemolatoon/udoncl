#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

size_t udononnx_matmul_nbits_cpu_gid;

#include "../../libraries/udononnxoperators/device/matmul_nbits.cl"

int main() {
  constexpr int words = 16384;
  constexpr int n = 7;
  constexpr int k = 113;
  std::vector<float> state(words, 0.0f);
  std::vector<float> output(words, 0.0f);
  std::vector<float> expected(n, 0.0f);
  for (int column = 0; column < n; ++column) {
    state[12416 + column] = static_cast<float>(column - 3) / 11.0f;
    expected[column] = state[12416 + column];
    for (int inner = 0; inner < k; ++inner) {
      int index = column * 128 + inner;
      state[index] = static_cast<float>((column * 5 + inner * 3) & 15);
      state[4096 + index] = static_cast<float>((inner / 32) + 1) / 19.0f;
      state[8192 + index] = 8.0f;
      state[12288 + inner] = static_cast<float>((inner % 17) - 8) / 13.0f;
      expected[column] += (state[index] - state[8192 + index]) *
                          state[4096 + index] * state[12288 + inner];
    }
  }
  for (udononnx_matmul_nbits_cpu_gid = 0; udononnx_matmul_nbits_cpu_gid < words;
       ++udononnx_matmul_nbits_cpu_gid)
    udononnx_matmul_nbits(state.data(), output.data(), n, k, 0, 1);
  float maximum_error = 0.0f;
  for (int column = 0; column < n; ++column)
    maximum_error = std::max(
        maximum_error, std::abs(output[12544 + column] - expected[column]));
  std::cout << "matmul_nbits_outputs=" << n << '\n';
  std::cout << "matmul_nbits_max_error=" << maximum_error << '\n';
  return maximum_error < 0.0001f ? 0 : 1;
}
