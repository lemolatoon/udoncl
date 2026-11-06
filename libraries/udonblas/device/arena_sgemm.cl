#ifdef __OPENCL_C_VERSION__
#define UDONBLAS_ARENA_X() ((size_t)get_global_id(0))
#define UDONBLAS_ARENA_Y() ((size_t)get_global_id(1))
#else
#include <stddef.h>
extern size_t udonblas_arena_sgemm_cpu_gid_x;
extern size_t udonblas_arena_sgemm_cpu_gid_y;
#define UDONBLAS_ARENA_X() (udonblas_arena_sgemm_cpu_gid_x)
#define UDONBLAS_ARENA_Y() (udonblas_arena_sgemm_cpu_gid_y)
#define __kernel
#define __global
#endif

#define UDONBLAS_ARENA_WIDTH 8192u

/* Private texture-arena backend for the public cuBLAS-shaped udonblasSgemm
 * call. ONNX tensors are row-major, so AOT supplies the validated transpose
 * flags and dimensions while one fragment owns one output element. */
__kernel void udonblas_arena_sgemm(__global const float *a,
                                   __global const float *b,
                                   __global float *output, int m, int n, int k,
                                   int transa, int transb, float alpha) {
  size_t gid = UDONBLAS_ARENA_Y() * UDONBLAS_ARENA_WIDTH + UDONBLAS_ARENA_X();
  size_t words = (size_t)m * (size_t)n;
  float value = 0.0f;
  if (gid < words) {
    int row = (int)(gid / (size_t)n);
    int column = (int)(gid % (size_t)n);
    float sum = 0.0f;
    for (int inner = 0; inner < 256; ++inner) {
      if (inner < k) {
        int a_index = transa ? inner * m + row : row * k + inner;
        int b_index = transb ? column * k + inner : inner * n + column;
        sum += a[(size_t)a_index & 16383u] * b[(size_t)b_index & 16383u];
      }
    }
    value = alpha * sum;
  }
  output[gid] = value;
}

/* Device-resident counterpart of udonblasSgemmStridedBatched. A zero input
 * stride broadcasts one matrix across every batch, matching the public API. */
__kernel void udonblas_arena_sgemm_strided_batched(
    __global const float *a, __global const float *b, __global float *output,
    int m, int n, int k, int stride_a, int stride_b, int stride_c,
    int batch_count, int transa, int transb, float alpha) {
  size_t gid = UDONBLAS_ARENA_Y() * UDONBLAS_ARENA_WIDTH + UDONBLAS_ARENA_X();
  size_t matrix_words = (size_t)m * (size_t)n;
  size_t words = (size_t)batch_count * matrix_words;
  float value = 0.0f;
  if (gid < words) {
    size_t batch = gid / matrix_words;
    size_t within = gid % matrix_words;
    int row = (int)(within / (size_t)n);
    int column = (int)(within % (size_t)n);
    size_t a_base = stride_a ? batch * (size_t)stride_a : 0u;
    size_t b_base = stride_b ? batch * (size_t)stride_b : 0u;
    (void)stride_c;
    float sum = 0.0f;
    for (int inner = 0; inner < 256; ++inner) {
      if (inner < k) {
        int a_index = transa ? inner * m + row : row * k + inner;
        int b_index = transb ? column * k + inner : inner * n + column;
        sum += a[(a_base + (size_t)a_index) & 16383u] *
               b[(b_base + (size_t)b_index) & 16383u];
      }
    }
    value = alpha * sum;
  }
  output[gid] = value;
}
