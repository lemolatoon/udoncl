#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

size_t udononnx_reindex_cpu_gid;

#include "../../libraries/udononnxoperators/device/reindex.cl"

static int packDimensions(const std::vector<int> &shape, int first) {
  unsigned low = first < static_cast<int>(shape.size()) ? shape[first] : 1;
  unsigned high =
      first + 1 < static_cast<int>(shape.size()) ? shape[first + 1] : 1;
  return static_cast<int>(low | (high << 16));
}

static int packPermutation(const std::vector<int> &permutation) {
  unsigned packed = 0;
  for (int axis = 0; axis < static_cast<int>(permutation.size()); ++axis)
    packed |= static_cast<unsigned>(permutation[axis]) << (axis * 3);
  return static_cast<int>(packed);
}

static float runCase(const std::vector<int> &shape,
                     const std::vector<int> &permutation) {
  constexpr int surfaceWords = 16384;
  int words = 1;
  for (int dimension : shape)
    words *= dimension;
  std::vector<float> input(words);
  std::vector<float> output(surfaceWords, -17.0f);
  std::vector<float> expected(words);
  for (int index = 0; index < words; ++index)
    input[index] = static_cast<float>((index * 13) % 37 - 18) / 11.0f;
  for (int outputIndex = 0; outputIndex < words; ++outputIndex) {
    int remaining = outputIndex;
    int inputIndex = 0;
    int inputStride = 1;
    std::vector<int> inputStrides(shape.size(), 1);
    for (int axis = static_cast<int>(shape.size()) - 1; axis >= 0; --axis) {
      inputStrides[axis] = inputStride;
      inputStride *= shape[axis];
    }
    for (int outputAxis = static_cast<int>(shape.size()) - 1; outputAxis >= 0;
         --outputAxis) {
      int inputAxis = permutation[outputAxis];
      int coordinate = remaining % shape[inputAxis];
      remaining /= shape[inputAxis];
      inputIndex += coordinate * inputStrides[inputAxis];
    }
    expected[outputIndex] = input[inputIndex];
  }
  for (udononnx_reindex_cpu_gid = 0; udononnx_reindex_cpu_gid < surfaceWords;
       ++udononnx_reindex_cpu_gid)
    udononnx_transpose_f32(
        input.data(), output.data(), words, static_cast<int>(shape.size()),
        packDimensions(shape, 0), packDimensions(shape, 2),
        packDimensions(shape, 4), packPermutation(permutation));
  float maximumError = 0.0f;
  for (int index = 0; index < words; ++index)
    maximumError =
        std::max(maximumError, std::abs(output[index] - expected[index]));
  return maximumError;
}

int main() {
  float headsError = runCase({1, 4, 32, 64}, {0, 2, 1, 3});
  float attentionError = runCase({1, 32, 4, 64}, {0, 1, 3, 2});
  float rotaryError = runCase({1, 32, 4}, {0, 2, 1});
  float maximumError = std::max({headsError, attentionError, rotaryError});
  std::cout << "reindex_cases=3\n";
  std::cout << "reindex_max_error=" << maximumError << '\n';
  return maximumError == 0.0f ? 0 : 1;
}
