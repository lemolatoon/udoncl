#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

size_t udononnx_matmul_nbits_cpu_gid;

#include "../../libraries/udononnxoperators/device/matmul_nbits.cl"

static float testHostManagedKernel() {
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
  return maximum_error;
}

static float testDeviceQ4Kernel(int m, int n, int k, const char *label) {
  constexpr int surface_words = 131072;
  int blocks = (k + 31) / 32;
  int pass_count = (k + 127) / 128;
  std::vector<float> activation(m * k);
  std::vector<uint> packed(n * blocks * 4, 0);
  std::vector<float> scales(n * blocks);
  std::vector<float> output(surface_words, -17.0f);
  std::vector<float> scratch(surface_words, 23.0f);
  std::vector<float> expected(m * n, 0.0f);

  for (int row = 0; row < m; ++row)
    for (int inner = 0; inner < k; ++inner)
      activation[row * k + inner] =
          static_cast<float>(((row + 3) * 11 + inner * 5) % 29 - 14) / 17.0f;
  for (int column = 0; column < n; ++column)
    for (int block = 0; block < blocks; ++block) {
      scales[column * blocks + block] =
          static_cast<float>((column + block) % 7 + 1) / 23.0f;
      for (int in_block = 0; in_block < 32; ++in_block) {
        int inner = block * 32 + in_block;
        if (inner >= k)
          break;
        uint quantized = static_cast<uint>((column * 3 + inner * 7) & 15);
        int word_index = (column * blocks + block) * 4 + in_block / 8;
        int shift = (in_block % 8) * 4;
        packed[word_index] |= quantized << shift;
        for (int row = 0; row < m; ++row)
          expected[row * n + column] += (static_cast<float>(quantized) - 8.0f) *
                                        scales[column * blocks + block] *
                                        activation[row * k + inner];
      }
    }

  float *previous = pass_count & 1 ? scratch.data() : output.data();
  float *destination = pass_count & 1 ? output.data() : scratch.data();
  for (int k_base = 0; k_base < k; k_base += 128) {
    int tile_k = std::min(128, k - k_base);
    for (udononnx_matmul_nbits_cpu_gid = 0;
         udononnx_matmul_nbits_cpu_gid < surface_words;
         ++udononnx_matmul_nbits_cpu_gid)
      udononnx_matmul_nbits_q4(activation.data(), packed.data(), scales.data(),
                               previous, destination, m, n, k, 32, 0, k_base, n,
                               tile_k, k_base != 0);
    std::swap(previous, destination);
  }

  if (previous != output.data())
    return INFINITY;
  float maximum_error = 0.0f;
  for (int index = 0; index < m * n; ++index)
    maximum_error =
        std::max(maximum_error, std::abs(output[index] - expected[index]));
  std::cout << "matmul_nbits_q4_case=" << label << " outputs=" << m * n
            << " dispatches=" << pass_count << " max_error=" << maximum_error
            << '\n';
  return maximum_error;
}

int main() {
  float host_error = testHostManagedKernel();
  float tail_error = testDeviceQ4Kernel(2, 37, 145, "tail");
  float tinyllama_k_error = testDeviceQ4Kernel(4, 33, 2048, "tinyllama_k");
  return host_error < 0.0001f && tail_error < 0.0001f &&
                 tinyllama_k_error < 0.0001f
             ? 0
             : 1;
}
