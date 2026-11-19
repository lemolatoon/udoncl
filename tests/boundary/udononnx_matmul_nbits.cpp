#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
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

/* The K-tiled schedule the one-dispatch kernel replaced: 128-wide K tiles
 * ping-ponged through previous/output with the same accumulation expression.
 * Kept here as the bit-equality reference. */
static void referenceQ4Tile(const float *activation, const uint *packed_weight,
                            const float *scales, const float *previous,
                            float *output, int m, int n, int k, int k_base,
                            int tile_k, int accumulate, uint gid) {
  float value = previous[gid];
  if (gid < static_cast<uint>(m * n)) {
    uint row = gid / static_cast<uint>(n);
    uint column = gid % static_cast<uint>(n);
    uint blocks = (static_cast<uint>(k) + 31u) / 32u;
    float sum = accumulate != 0 ? previous[gid] : 0.0f;
    for (uint inner = 0; inner < 128u; ++inner) {
      uint inner_index = static_cast<uint>(k_base) + inner;
      if (inner < static_cast<uint>(tile_k) &&
          inner_index < static_cast<uint>(k)) {
        uint block = inner_index / 32u;
        uint in_block = inner_index % 32u;
        uint word_index = (column * blocks + block) * 4u + in_block / 8u;
        uint shift = (in_block % 8u) * 4u;
        uint quantized = (packed_weight[word_index] >> shift) & 15u;
        float weight = (static_cast<float>(quantized) - 8.0f) *
                       scales[column * blocks + block];
        sum += weight * activation[row * static_cast<uint>(k) + inner_index];
      }
    }
    value = sum;
  }
  output[gid] = value;
}

struct Q4Case {
  int m, n, k;
  std::vector<float> activation;
  std::vector<uint> packed;
  std::vector<float> scales;
  std::vector<double> expected;
};

static Q4Case makeQ4Case(int m, int n, int k) {
  Q4Case data{m, n, k, {}, {}, {}, {}};
  int blocks = (k + 31) / 32;
  data.activation.resize(m * k);
  data.packed.assign(n * blocks * 4, 0);
  data.scales.resize(n * blocks);
  data.expected.assign(m * n, 0.0);
  for (int row = 0; row < m; ++row)
    for (int inner = 0; inner < k; ++inner)
      data.activation[row * k + inner] =
          static_cast<float>(((row + 3) * 11 + inner * 5) % 29 - 14) / 17.0f;
  for (int column = 0; column < n; ++column)
    for (int block = 0; block < blocks; ++block) {
      data.scales[column * blocks + block] =
          static_cast<float>((column + block) % 7 + 1) / 23.0f;
      for (int in_block = 0; in_block < 32; ++in_block) {
        int inner = block * 32 + in_block;
        if (inner >= k)
          break;
        uint quantized = static_cast<uint>((column * 3 + inner * 7) & 15);
        int word_index = (column * blocks + block) * 4 + in_block / 8;
        data.packed[word_index] |= quantized << ((in_block % 8) * 4);
        for (int row = 0; row < m; ++row)
          data.expected[row * n + column] +=
              (static_cast<double>(quantized) - 8.0) *
              data.scales[column * blocks + block] *
              data.activation[row * k + inner];
      }
    }
  return data;
}

constexpr int kQ4SurfaceWords = 131072;

static std::vector<float> runReference(const Q4Case &data) {
  int passes = (data.k + 127) / 128;
  std::vector<float> output(kQ4SurfaceWords, -17.0f);
  std::vector<float> scratch(kQ4SurfaceWords, 23.0f);
  float *previous = passes & 1 ? scratch.data() : output.data();
  float *destination = passes & 1 ? output.data() : scratch.data();
  for (int k_base = 0; k_base < data.k; k_base += 128) {
    for (uint gid = 0; gid < static_cast<uint>(kQ4SurfaceWords); ++gid)
      referenceQ4Tile(data.activation.data(), data.packed.data(),
                      data.scales.data(), previous, destination, data.m, data.n,
                      data.k, k_base, std::min(128, data.k - k_base),
                      k_base != 0, gid);
    std::swap(previous, destination);
  }
  return std::vector<float>(previous, previous + data.m * data.n);
}

/* Runs udononnx_matmul_nbits_q4 over consecutive K-block segments,
 * ping-ponging previous/output exactly like the Host lowering so the last
 * segment writes output. Returns false when a word outside the m x n output
 * is not zero. */
static bool runOnePass(const Q4Case &data, const std::vector<int> &segments,
                       std::vector<float> &result) {
  std::vector<float> output(kQ4SurfaceWords, -17.0f);
  std::vector<float> scratch(kQ4SurfaceWords, 23.0f);
  int count = static_cast<int>(segments.size());
  int block_base = 0;
  for (int segment = 0; segment < count; ++segment) {
    bool writes_output = (count - 1 - segment) % 2 == 0;
    float *destination = writes_output ? output.data() : scratch.data();
    float *previous = writes_output ? scratch.data() : output.data();
    for (udononnx_matmul_nbits_cpu_gid = 0;
         udononnx_matmul_nbits_cpu_gid < kQ4SurfaceWords;
         ++udononnx_matmul_nbits_cpu_gid)
      udononnx_matmul_nbits_q4(data.activation.data(), data.packed.data(),
                               data.scales.data(), previous, destination,
                               data.m, data.n, data.k, block_base,
                               segments[segment], segment != 0);
    block_base += segments[segment];
  }
  result.assign(output.begin(), output.begin() + data.m * data.n);
  for (int index = data.m * data.n; index < kQ4SurfaceWords; ++index)
    if (output[index] != 0.0f)
      return false;
  return true;
}

/* parts == 0 splits the K blocks like the Host lowering:
 * ceil(blocks / UDONONNX_Q4_DISPATCH_BLOCKS) segments, each full but the
 * last. Otherwise `parts` nearly equal segments exercise the partial-sum
 * hand-off at arbitrary block boundaries. */
static std::vector<int> q4Segments(int k, int parts) {
  int blocks = (k + 31) / 32;
  int limit = static_cast<int>(UDONONNX_Q4_DISPATCH_BLOCKS);
  std::vector<int> segments;
  if (parts == 0) {
    for (int base = 0; base < blocks; base += limit)
      segments.push_back(std::min(limit, blocks - base));
    return segments;
  }
  for (int part = 0; part < parts; ++part)
    segments.push_back(blocks * (part + 1) / parts - blocks * part / parts);
  return segments;
}

static bool testDeviceQ4Kernel(int m, int n, int k) {
  Q4Case data = makeQ4Case(m, n, k);
  std::vector<float> reference = runReference(data);
  double maximum_error = 0.0;
  for (int index = 0; index < m * n; ++index)
    maximum_error = std::max(maximum_error,
                             std::abs(reference[index] - data.expected[index]));
  bool passed = maximum_error < 0.0001;
  int blocks = (k + 31) / 32;
  int host_segments = static_cast<int>(q4Segments(k, 0).size());
  std::cout << "matmul_nbits_q4 m=" << m << " n=" << n << " k=" << k
            << " tiled_dispatches=" << (k + 127) / 128
            << " dispatches=" << host_segments
            << " max_error=" << maximum_error;
  for (int parts : {0, 1, 2, 3}) {
    if (parts > blocks ||
        static_cast<int64_t>(blocks) >
            static_cast<int64_t>(parts ? parts : host_segments) *
                static_cast<int64_t>(UDONONNX_Q4_DISPATCH_BLOCKS))
      continue;
    std::vector<float> result;
    bool zero_padding = runOnePass(data, q4Segments(k, parts), result);
    int mismatches = 0;
    for (int index = 0; index < m * n; ++index)
      mismatches +=
          std::memcmp(&result[index], &reference[index], sizeof(float)) != 0;
    std::cout << " split" << parts << "_mismatches=" << mismatches;
    passed = passed && zero_padding && mismatches == 0;
  }
  std::cout << (passed ? " passed" : " FAILED") << '\n';
  return passed;
}

int main() {
  float host_error = testHostManagedKernel();
  bool q4_passed = true;
  /* K covers one block, K tails, one 128-wide tile, the 16-block chunk and
   * 256-block dispatch boundaries, and K that needs two and three dispatch
   * segments; n and m cover single words, tails, and up to 4 rows. */
  const int shapes[][3] = {{1, 1, 32},    {4, 1, 32},     {3, 7, 31},
                           {2, 37, 145},  {1, 2048, 128}, {2, 5, 544},
                           {4, 33, 2048}, {1, 3, 5632},   {4, 5, 5632},
                           {1, 3, 8192},  {2, 3, 8225},   {1, 2, 16416}};
  for (const auto &shape : shapes)
    q4_passed = testDeviceQ4Kernel(shape[0], shape[1], shape[2]) && q4_passed;
  return host_error < 0.0001f && q4_passed ? 0 : 1;
}
