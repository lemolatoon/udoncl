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

static int packSigned(const std::vector<int> &values, int first, int fallback) {
  unsigned low = static_cast<unsigned>(static_cast<unsigned short>(
      first < static_cast<int>(values.size()) ? values[first] : fallback));
  unsigned high = static_cast<unsigned>(static_cast<unsigned short>(
      first + 1 < static_cast<int>(values.size()) ? values[first + 1]
                                                  : fallback));
  return static_cast<int>(low | (high << 16));
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

static float runSliceCase(const std::vector<int> &inputShape,
                          const std::vector<int> &outputShape,
                          const std::vector<int> &starts,
                          const std::vector<int> &steps) {
  constexpr int surfaceWords = 16384;
  int inputWords = 1;
  int outputWords = 1;
  for (int dimension : inputShape)
    inputWords *= dimension;
  for (int dimension : outputShape)
    outputWords *= dimension;
  std::vector<float> input(inputWords);
  std::vector<float> output(surfaceWords, -17.0f);
  std::vector<float> expected(outputWords);
  for (int index = 0; index < inputWords; ++index)
    input[index] = static_cast<float>((index * 17) % 43 - 21) / 7.0f;
  for (int outputIndex = 0; outputIndex < outputWords; ++outputIndex) {
    int remaining = outputIndex;
    int inputIndex = 0;
    int inputStride = 1;
    for (int axis = static_cast<int>(inputShape.size()) - 1; axis >= 0;
         --axis) {
      int coordinate = remaining % outputShape[axis];
      remaining /= outputShape[axis];
      inputIndex += (starts[axis] + coordinate * steps[axis]) * inputStride;
      inputStride *= inputShape[axis];
    }
    expected[outputIndex] = input[inputIndex];
  }
  for (udononnx_reindex_cpu_gid = 0; udononnx_reindex_cpu_gid < surfaceWords;
       ++udononnx_reindex_cpu_gid)
    udononnx_slice_f32(
        input.data(), output.data(), outputWords,
        static_cast<int>(inputShape.size()), packDimensions(inputShape, 0),
        packDimensions(inputShape, 2), packDimensions(inputShape, 4),
        packDimensions(outputShape, 0), packDimensions(outputShape, 2),
        packDimensions(outputShape, 4), packSigned(starts, 0, 0),
        packSigned(starts, 2, 0), packSigned(starts, 4, 0),
        packSigned(steps, 0, 1), packSigned(steps, 2, 1),
        packSigned(steps, 4, 1));
  float maximumError = 0.0f;
  for (int index = 0; index < outputWords; ++index)
    maximumError =
        std::max(maximumError, std::abs(output[index] - expected[index]));
  return maximumError;
}

static float runExpandCase(const std::vector<int> &inputShape,
                           const std::vector<int> &outputShape) {
  constexpr int surfaceWords = 16384;
  int inputWords = 1;
  int outputWords = 1;
  for (int dimension : inputShape)
    inputWords *= dimension;
  for (int dimension : outputShape)
    outputWords *= dimension;
  std::vector<float> input(inputWords);
  std::vector<float> output(surfaceWords, -17.0f);
  std::vector<float> expected(outputWords);
  for (int index = 0; index < inputWords; ++index)
    input[index] = static_cast<float>((index * 19) % 47 - 23) / 9.0f;
  int rankOffset = static_cast<int>(outputShape.size()) -
                   static_cast<int>(inputShape.size());
  for (int outputIndex = 0; outputIndex < outputWords; ++outputIndex) {
    int remaining = outputIndex;
    int inputIndex = 0;
    int inputStride = 1;
    for (int outputAxis = static_cast<int>(outputShape.size()) - 1;
         outputAxis >= 0; --outputAxis) {
      int coordinate = remaining % outputShape[outputAxis];
      remaining /= outputShape[outputAxis];
      int inputAxis = outputAxis - rankOffset;
      if (inputAxis >= 0) {
        if (inputShape[inputAxis] != 1)
          inputIndex += coordinate * inputStride;
        inputStride *= inputShape[inputAxis];
      }
    }
    expected[outputIndex] = input[inputIndex];
  }
  for (udononnx_reindex_cpu_gid = 0; udononnx_reindex_cpu_gid < surfaceWords;
       ++udononnx_reindex_cpu_gid)
    udononnx_expand_f32(
        input.data(), output.data(), outputWords,
        static_cast<int>(inputShape.size()), packDimensions(inputShape, 0),
        packDimensions(inputShape, 2), packDimensions(inputShape, 4),
        static_cast<int>(outputShape.size()), packDimensions(outputShape, 0),
        packDimensions(outputShape, 2), packDimensions(outputShape, 4));
  float maximumError = 0.0f;
  for (int index = 0; index < outputWords; ++index)
    maximumError =
        std::max(maximumError, std::abs(output[index] - expected[index]));
  return maximumError;
}

int main() {
  float headsError = runCase({1, 4, 32, 64}, {0, 2, 1, 3});
  float attentionError = runCase({1, 32, 4, 64}, {0, 1, 3, 2});
  float rotaryError = runCase({1, 32, 4}, {0, 2, 1});
  float maximumError = std::max({headsError, attentionError, rotaryError});
  float halfError =
      runSliceCase({1, 32, 4, 64}, {1, 32, 4, 32}, {0, 0, 0, 32}, {1, 1, 1, 1});
  float stridedError = runSliceCase({2, 3, 4}, {2, 3, 2}, {0, 0, 0}, {1, 1, 2});
  float repeatHeadsError = runExpandCase({1, 4, 1, 4, 64}, {1, 4, 8, 4, 64});
  float leadingBroadcastError = runExpandCase({2, 1}, {3, 2, 5});
  maximumError = std::max({maximumError, halfError, stridedError,
                           repeatHeadsError, leadingBroadcastError});
  std::cout << "reindex_cases=7\n";
  std::cout << "reindex_max_error=" << maximumError << '\n';
  return maximumError == 0.0f ? 0 : 1;
}
