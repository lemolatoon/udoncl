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
  /* Sum of |term| of each output: the scale of its rounding error. */
  std::vector<double> magnitude;
};

static Q4Case makeQ4Case(int m, int n, int k) {
  Q4Case data{m, n, k, {}, {}, {}, {}, {}};
  int blocks = (k + 31) / 32;
  data.activation.resize(m * k);
  data.packed.assign(n * blocks * 4, 0);
  data.scales.resize(n * blocks);
  data.expected.assign(m * n, 0.0);
  data.magnitude.assign(m * n, 0.0);
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
        for (int row = 0; row < m; ++row) {
          double term = (static_cast<double>(quantized) - 8.0) *
                        data.scales[column * blocks + block] *
                        data.activation[row * k + inner];
          data.expected[row * n + column] += term;
          data.magnitude[row * n + column] += std::abs(term);
        }
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

/* Runs udononnx_matmul_nbits_q4 like the Host lowering: one dispatch for a
 * single split, otherwise a partial pass of `splits` planes of `block_count`
 * blocks into the scratch surface and an ordered sum pass into the output.
 * Returns false when a word outside the written planes is not zero. */
static bool runSchedule(const Q4Case &data, int block_count, int splits,
                        std::vector<float> &result) {
  std::vector<float> output(kQ4SurfaceWords, -17.0f);
  std::vector<float> scratch(kQ4SurfaceWords, 23.0f);
  auto pass = [&](const float *previous, float *destination, int count,
                  int planes, int partials) {
    for (udononnx_matmul_nbits_cpu_gid = 0;
         udononnx_matmul_nbits_cpu_gid < kQ4SurfaceWords;
         ++udononnx_matmul_nbits_cpu_gid)
      udononnx_matmul_nbits_q4(data.activation.data(), data.packed.data(),
                               data.scales.data(), previous, destination,
                               data.m, data.n, data.k, count, planes, partials);
  };
  bool zero_padding = true;
  if (splits == 1) {
    pass(scratch.data(), output.data(), block_count, 1, 0);
  } else {
    pass(output.data(), scratch.data(), block_count, splits, 0);
    for (int index = splits * data.m * data.n; index < kQ4SurfaceWords; ++index)
      zero_padding = zero_padding && scratch[index] == 0.0f;
    pass(scratch.data(), output.data(), 0, 1, splits);
  }
  result.assign(output.begin(), output.begin() + data.m * data.n);
  for (int index = data.m * data.n; index < kQ4SurfaceWords; ++index)
    zero_padding = zero_padding && output[index] == 0.0f;
  return zero_padding;
}

/* The documented split-K order, written independently of the kernel: each
 * split sums its blocks element by element in ascending k with the kernel's
 * expression, and the partials are added in ascending split order starting
 * from 0. */
static std::vector<float> splitReference(const Q4Case &data, int block_count,
                                         int splits) {
  int blocks = (data.k + 31) / 32;
  std::vector<float> result(data.m * data.n, 0.0f);
  for (int row = 0; row < data.m; ++row)
    for (int column = 0; column < data.n; ++column) {
      float total = 0.0f;
      for (int split = 0; split < splits; ++split) {
        float partial = 0.0f;
        for (int block = split * block_count;
             block < std::min(blocks, (split + 1) * block_count); ++block) {
          float scale = data.scales[column * blocks + block];
          for (int in_block = 0; in_block < 32; ++in_block) {
            int inner = block * 32 + in_block;
            if (inner >= data.k)
              break;
            uint packed =
                data.packed[(column * blocks + block) * 4 + in_block / 8];
            partial +=
                ((static_cast<float>((packed >> ((in_block % 8) * 4)) & 15u) -
                  8.0f) *
                 scale) *
                data.activation[row * data.k + inner];
          }
        }
        total += partial;
      }
      result[row * data.n + column] = total;
    }
  return result;
}

static int mismatchesOf(const std::vector<float> &left,
                        const std::vector<float> &right) {
  int mismatches = 0;
  for (size_t index = 0; index < left.size(); ++index)
    mismatches += std::memcmp(&left[index], &right[index], sizeof(float)) != 0;
  return mismatches;
}

/* Largest |result - expected| relative to the output's sum of |term|. */
static double relativeError(const Q4Case &data,
                            const std::vector<float> &result) {
  double error = 0.0;
  for (size_t index = 0; index < result.size(); ++index)
    error = std::max(error, std::abs(result[index] - data.expected[index]) /
                                std::max(1.0, data.magnitude[index]));
  return error;
}

/* Split counts to exercise: the serial schedule when one fragment covers K,
 * then two, three, seven and the finest split the kernel and the
 * 131072-word partial surface allow, each normalized so no split is empty. */
static std::vector<int> splitCounts(int m, int n, int blocks) {
  std::vector<int> counts;
  for (int wanted : {1, 2, 3, 7, 64}) {
    int splits =
        std::min({wanted, blocks, static_cast<int>(UDONONNX_Q4_MAX_SPLITS),
                  kQ4SurfaceWords / (m * n)});
    int block_count = (blocks + splits - 1) / splits;
    splits = (blocks + block_count - 1) / block_count;
    if (block_count > static_cast<int>(UDONONNX_Q4_DISPATCH_BLOCKS) ||
        std::find(counts.begin(), counts.end(), splits) != counts.end())
      continue;
    counts.push_back(splits);
  }
  return counts;
}

/* Split-K changes the summation order, so a split result is not compared bit
 * for bit with the serial order. Instead it must
 *  - match the independent reference of the documented order bit for bit,
 *  - be bit identical when the schedule runs again (determinism), and
 *  - stay within the float rounding bound of that order against the double
 *    product: gamma(n) = n u / (1 - n u) of the output's sum of |term|, with
 *    n = 32 * block_count + splits + 2 dependent roundings and u = 2^-24.
 * One split must stay bit identical to the serial K-tiled reference. */
static bool testDeviceQ4Kernel(int m, int n, int k) {
  Q4Case data = makeQ4Case(m, n, k);
  std::vector<float> reference = runReference(data);
  double maximum_error = 0.0;
  for (int index = 0; index < m * n; ++index)
    maximum_error = std::max(maximum_error,
                             std::abs(reference[index] - data.expected[index]));
  bool passed = maximum_error < 0.0001;
  int blocks = (k + 31) / 32;
  std::cout << "matmul_nbits_q4 m=" << m << " n=" << n << " k=" << k
            << " tiled_dispatches=" << (k + 127) / 128
            << " serial_max_error=" << maximum_error
            << " serial_relative_error=" << relativeError(data, reference);
  for (int splits : splitCounts(m, n, blocks)) {
    int block_count = (blocks + splits - 1) / splits;
    std::vector<float> result;
    std::vector<float> repeated;
    bool zero_padding = runSchedule(data, block_count, splits, result);
    zero_padding =
        runSchedule(data, block_count, splits, repeated) && zero_padding;
    int nondeterministic = mismatchesOf(result, repeated);
    int order_mismatches =
        mismatchesOf(result, splitReference(data, block_count, splits));
    int serial_mismatches = mismatchesOf(result, reference);
    double error = relativeError(data, result);
    double roundings = 32.0 * block_count + splits + 2.0;
    double unit = std::ldexp(1.0, -24);
    double tolerance = roundings * unit / (1.0 - roundings * unit);
    bool split_passed = zero_padding && nondeterministic == 0 &&
                        order_mismatches == 0 && error <= tolerance &&
                        (splits != 1 || serial_mismatches == 0);
    std::cout << " split" << splits << "x" << block_count
              << ":relative_error=" << error << "/tolerance=" << tolerance
              << "/order_mismatches=" << order_mismatches
              << "/repeat_mismatches=" << nondeterministic
              << "/serial_mismatches=" << serial_mismatches
              << (split_passed ? "" : "/FAILED");
    passed = passed && split_passed;
  }
  /* previous_partials and a K slice in one dispatch: two planes of previous
   * are added, in order, before the whole-K sum. */
  if (blocks <= static_cast<int>(UDONONNX_Q4_DISPATCH_BLOCKS) &&
      2 * m * n <= kQ4SurfaceWords) {
    std::vector<float> previous(kQ4SurfaceWords, 0.0f);
    std::vector<float> output(kQ4SurfaceWords, -17.0f);
    for (int index = 0; index < 2 * m * n; ++index)
      previous[index] = static_cast<float>(index % 13 - 6) / 7.0f;
    for (udononnx_matmul_nbits_cpu_gid = 0;
         udononnx_matmul_nbits_cpu_gid < kQ4SurfaceWords;
         ++udononnx_matmul_nbits_cpu_gid)
      udononnx_matmul_nbits_q4(data.activation.data(), data.packed.data(),
                               data.scales.data(), previous.data(),
                               output.data(), m, n, k, blocks, 1, 2);
    int mismatches = 0;
    for (int row = 0; row < m; ++row)
      for (int column = 0; column < n; ++column) {
        int word = row * n + column;
        /* The kernel's order: previous planes first, then the K blocks
         * element by element, all in one float accumulator. */
        float sum = (0.0f + previous[word]) + previous[m * n + word];
        for (int inner = 0; inner < k; ++inner) {
          int block = inner / 32;
          uint packed =
              data.packed[(column * blocks + block) * 4 + (inner % 32) / 8];
          sum += ((static_cast<float>((packed >> ((inner % 8) * 4)) & 15u) -
                   8.0f) *
                  data.scales[column * blocks + block]) *
                 data.activation[row * k + inner];
        }
        mismatches += std::memcmp(&sum, &output[word], sizeof(float)) != 0;
      }
    std::cout << " previous_partials_mismatches=" << mismatches;
    passed = passed && mismatches == 0;
  }
  std::cout << (passed ? " passed" : " FAILED") << '\n';
  return passed;
}

/* udoncl-merge-onnx-sibling-matmulnbits stacks the weight and scale rows of
 * MatMulNBits nodes that read one activation and slices the result. Under
 * the same split schedule every output word must be bit-identical to the
 * separate nodes'. (The Host may choose a different schedule for the merged
 * node; the results then agree within rounding.) */
static Q4Case columnsOf(const Q4Case &stacked, int first, int count) {
  int blocks = (stacked.k + 31) / 32;
  Q4Case part{stacked.m, count, stacked.k, stacked.activation, {}, {}, {}, {}};
  part.packed.assign(stacked.packed.begin() + first * blocks * 4,
                     stacked.packed.begin() + (first + count) * blocks * 4);
  part.scales.assign(stacked.scales.begin() + first * blocks,
                     stacked.scales.begin() + (first + count) * blocks);
  return part;
}

static bool testStackedColumns(int m, int k, std::vector<int> widths,
                               int splits) {
  int n = 0;
  for (int width : widths)
    n += width;
  int blocks = (k + 31) / 32;
  int block_count = (blocks + splits - 1) / splits;
  splits = (blocks + block_count - 1) / block_count;
  Q4Case stacked = makeQ4Case(m, n, k);
  std::vector<float> merged;
  bool passed = runSchedule(stacked, block_count, splits, merged);
  int mismatches = 0;
  int first = 0;
  for (int width : widths) {
    std::vector<float> separate;
    passed = runSchedule(columnsOf(stacked, first, width), block_count, splits,
                         separate) &&
             passed;
    for (int row = 0; row < m; ++row)
      for (int column = 0; column < width; ++column)
        mismatches +=
            std::memcmp(&merged[row * n + first + column],
                        &separate[row * width + column], sizeof(float)) != 0;
    first += width;
  }
  passed = passed && mismatches == 0;
  std::cout << "matmul_nbits_q4_stacked m=" << m << " k=" << k << " n=" << n
            << " parts=" << widths.size() << " splits=" << splits
            << " mismatches=" << mismatches << (passed ? " passed" : " FAILED")
            << '\n';
  return passed;
}

int main() {
  float host_error = testHostManagedKernel();
  bool stacked_passed = testStackedColumns(1, 2048, {64, 8, 8}, 1);
  stacked_passed =
      testStackedColumns(1, 2048, {64, 8, 8}, 16) && stacked_passed;
  stacked_passed = testStackedColumns(3, 145, {5, 1, 7}, 3) && stacked_passed;
  stacked_passed = testStackedColumns(2, 8225, {3, 2}, 2) && stacked_passed;
  bool q4_passed = true;
  /* K covers one block, K tails, one 128-wide tile, the 16-block chunk and
   * 256-block slice boundaries, and K whose blocks need two and three
   * splits; n and m cover single words, tails, and up to 4 rows. */
  const int shapes[][3] = {{1, 1, 32},    {4, 1, 32},     {3, 7, 31},
                           {2, 37, 145},  {1, 2048, 128}, {2, 5, 544},
                           {4, 33, 2048}, {1, 3, 5632},   {4, 5, 5632},
                           {1, 3, 8192},  {2, 3, 8225},   {1, 2, 16416}};
  for (const auto &shape : shapes)
    q4_passed = testDeviceQ4Kernel(shape[0], shape[1], shape[2]) && q4_passed;
  return host_error < 0.0001f && q4_passed && stacked_passed ? 0 : 1;
}
