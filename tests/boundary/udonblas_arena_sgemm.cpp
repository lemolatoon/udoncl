#include <algorithm>
#include <cmath>
#include <cstddef>
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

int main() {
  float attentionError = runCase(32, 4, 64, 4, false);
  float broadcastError = runCase(2, 2, 4, 3, true);
  std::cout << "arena_sgemm_attention_outputs=" << 32 * 4 * 64 << '\n';
  std::cout << "arena_sgemm_broadcast_outputs=" << 2 * 2 * 4 << '\n';
  std::cout << "arena_sgemm_max_error="
            << std::max(attentionError, broadcastError) << '\n';
  return attentionError < 0.0001f && broadcastError < 0.0001f ? 0 : 1;
}
