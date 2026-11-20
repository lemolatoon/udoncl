#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <vector>

size_t udonblas_arena_sgemm_cpu_gid_x;
size_t udonblas_arena_sgemm_cpu_gid_y;

#include "../../libraries/udonblas/device/arena_sgemm.cl"

static float value(int index, int salt) {
  return static_cast<float>(((index + 3) * (salt * 7 + 11)) % 31 - 15) / 19.0f;
}

static float runCase(int batches, int m, int n, int k, bool broadcastB) {
  constexpr int surfaceWords = 16384;
  int strideA = m * k;
  int strideB = broadcastB ? 0 : k * n;
  int strideC = m * n;
  std::vector<float> a(batches * strideA);
  std::vector<float> b((broadcastB ? 1 : batches) * k * n);
  std::vector<float> output(surfaceWords, -37.0f);
  std::vector<float> expected(batches * strideC, 0.0f);
  for (int index = 0; index < static_cast<int>(a.size()); ++index)
    a[index] = value(index, 2);
  for (int index = 0; index < static_cast<int>(b.size()); ++index)
    b[index] = value(index, 5);
  for (int batch = 0; batch < batches; ++batch)
    for (int row = 0; row < m; ++row)
      for (int column = 0; column < n; ++column)
        for (int inner = 0; inner < k; ++inner)
          expected[batch * strideC + row * n + column] +=
              a[batch * strideA + row * k + inner] *
              b[(broadcastB ? 0 : batch * k * n) + inner * n + column];
  for (udonblas_arena_sgemm_cpu_gid_y = 0; udonblas_arena_sgemm_cpu_gid_y < 2;
       ++udonblas_arena_sgemm_cpu_gid_y)
    for (udonblas_arena_sgemm_cpu_gid_x = 0;
         udonblas_arena_sgemm_cpu_gid_x < 8192;
         ++udonblas_arena_sgemm_cpu_gid_x)
      udonblas_arena_sgemm_strided_batched(a.data(), b.data(), output.data(), m,
                                           n, k, strideA, strideB, strideC,
                                           batches, 0, 0, 1.0f);
  float maximumError = 0.0f;
  for (int index = 0; index < static_cast<int>(expected.size()); ++index)
    maximumError =
        std::max(maximumError, std::abs(output[index] - expected[index]));
  return maximumError;
}

/* Non-batched kernel over transposed or plain row-major operands. K beyond
 * one 256-wide chunk exercises the guarded chunk loop. */
static float runPlainCase(int m, int n, int k, int transa, int transb) {
  constexpr int surfaceWords = 16384;
  std::vector<float> a(m * k);
  std::vector<float> b(k * n);
  std::vector<float> output(surfaceWords, -37.0f);
  std::vector<double> expected(m * n, 0.0);
  std::vector<double> magnitude(m * n, 0.0);
  for (int index = 0; index < m * k; ++index)
    a[index] = value(index, 3);
  for (int index = 0; index < k * n; ++index)
    b[index] = value(index, 4);
  for (int row = 0; row < m; ++row)
    for (int column = 0; column < n; ++column)
      for (int inner = 0; inner < k; ++inner) {
        double term = static_cast<double>(transa ? a[inner * m + row]
                                                 : a[row * k + inner]) *
                      (transb ? b[column * k + inner] : b[inner * n + column]);
        expected[row * n + column] += term;
        magnitude[row * n + column] += std::abs(term);
      }
  for (udonblas_arena_sgemm_cpu_gid_y = 0; udonblas_arena_sgemm_cpu_gid_y < 2;
       ++udonblas_arena_sgemm_cpu_gid_y)
    for (udonblas_arena_sgemm_cpu_gid_x = 0;
         udonblas_arena_sgemm_cpu_gid_x < 8192;
         ++udonblas_arena_sgemm_cpu_gid_x)
      udonblas_arena_sgemm(a.data(), b.data(), output.data(), m, n, k, transa,
                           transb, 1.0f);
  double maximumError = 0.0;
  /* Error relative to the sum of term magnitudes, which bounds FP32
   * accumulation error independently of K. */
  for (int index = 0; index < m * n; ++index)
    maximumError =
        std::max(maximumError, std::abs(output[index] - expected[index]) /
                                   (1.0 + magnitude[index]));
  for (int index = m * n; index < surfaceWords; ++index)
    if (output[index] != 0.0f)
      return INFINITY;
  return static_cast<float>(maximumError);
}

static std::vector<float> runBatched(const std::vector<float> &a,
                                     const std::vector<float> &b, int batches,
                                     int m, int n, int k) {
  std::vector<float> output(131072, -37.0f);
  for (udonblas_arena_sgemm_cpu_gid_y = 0; udonblas_arena_sgemm_cpu_gid_y < 16;
       ++udonblas_arena_sgemm_cpu_gid_y)
    for (udonblas_arena_sgemm_cpu_gid_x = 0;
         udonblas_arena_sgemm_cpu_gid_x < 8192;
         ++udonblas_arena_sgemm_cpu_gid_x)
      udonblas_arena_sgemm_strided_batched(a.data(), b.data(), output.data(), m,
                                           n, k, m * k, k * n, m * n, batches,
                                           0, 0, 1.0f);
  output.resize(batches * m * n);
  return output;
}

/* udoncl-fold-onnx-repeated-matmul-operand turns G * R batches of M rows
 * whose right matrices repeat R times into G batches of R * M rows over the
 * distinct matrices. Every output word must stay bit-identical. */
static int runRepeatedOperandCase(int groups, int repeats, int m, int n,
                                  int k) {
  std::vector<float> a(groups * repeats * m * k);
  std::vector<float> distinct(groups * k * n);
  std::vector<float> repeated;
  for (int index = 0; index < static_cast<int>(a.size()); ++index)
    a[index] = value(index, 6) * (index % 5 == 0 ? 1.0e-3f : 1.0f);
  for (int index = 0; index < static_cast<int>(distinct.size()); ++index)
    distinct[index] = value(index, 8) * (index % 7 == 0 ? 3.0e4f : 1.0f);
  for (int group = 0; group < groups; ++group)
    for (int repeat = 0; repeat < repeats; ++repeat)
      repeated.insert(repeated.end(), distinct.begin() + group * k * n,
                      distinct.begin() + (group + 1) * k * n);
  std::vector<float> expanded =
      runBatched(a, repeated, groups * repeats, m, n, k);
  std::vector<float> folded =
      runBatched(a, distinct, groups, repeats * m, n, k);
  int mismatches = 0;
  for (size_t index = 0; index < expanded.size(); ++index)
    mismatches +=
        std::memcmp(&expanded[index], &folded[index], sizeof(float)) != 0;
  std::cout << "arena_sgemm_repeated_operand groups=" << groups
            << " repeats=" << repeats << " m=" << m
            << " mismatches=" << mismatches << '\n';
  return mismatches;
}

int main() {
  int repeatedMismatches = runRepeatedOperandCase(4, 8, 1, 64, 64) +
                           runRepeatedOperandCase(2, 3, 2, 5, 300);
  float attentionError = runCase(32, 4, 64, 4, false);
  float m1AttentionError = runCase(32, 1, 64, 64, false);
  float broadcastError = runCase(2, 2, 4, 3, true);
  float longKBatchedError = runCase(1, 2, 3, 1000, false);
  float longKError =
      std::max({runPlainCase(3, 5, 600, 0, 0), runPlainCase(4, 3, 1027, 1, 0),
                runPlainCase(2, 2, 8192, 0, 1)});
  std::cout << "arena_sgemm_long_k_max_error="
            << std::max(longKBatchedError, longKError) << '\n';
  std::cout << "arena_sgemm_attention_outputs=" << 32 * 4 * 64 << '\n';
  std::cout << "arena_sgemm_m1_attention_outputs=" << 32 * 1 * 64 << '\n';
  std::cout << "arena_sgemm_broadcast_outputs=" << 2 * 2 * 4 << '\n';
  std::cout << "arena_sgemm_max_error="
            << std::max({attentionError, m1AttentionError, broadcastError})
            << '\n';
  return repeatedMismatches == 0 && attentionError < 0.0001f &&
                 m1AttentionError < 0.0001f && broadcastError < 0.0001f &&
                 longKBatchedError < 0.001f && longKError < 0.0001f
             ? 0
             : 1;
}
