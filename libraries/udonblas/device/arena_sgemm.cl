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
/* The Device lowering accepts only constant loop bounds. K is walked in
 * guarded chunks so a short K skips whole chunks with one compare each; the
 * accumulation order stays strictly ascending in K. A chunk of 64 lanes is a
 * short counted loop that the HLSL emitter unrolls, so the Loads of a chunk
 * issue back to back; a K of 64 (an attention head) is exactly one chunk. */
#define UDONBLAS_ARENA_K_CHUNK 64
#define UDONBLAS_ARENA_K_CHUNKS 128 /* K <= 8192 */

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
    for (int chunk = 0; chunk < UDONBLAS_ARENA_K_CHUNKS; ++chunk)
      if (chunk * UDONBLAS_ARENA_K_CHUNK < k)
        for (int lane = 0; lane < UDONBLAS_ARENA_K_CHUNK; ++lane) {
          int inner = chunk * UDONBLAS_ARENA_K_CHUNK + lane;
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
 * stride broadcasts one matrix across every batch, matching the public API.
 * OPERAND_MASK bounds every operand address for the gather proof: one less
 * than a power of two that holds the largest operand the kernel admits.
 *
 * k_begin is private to the Device backend and 0 for the public call: the
 * terms whose inner index is below it are not accumulated. ONNX Host
 * finalization passes a larger value only where it proves those terms are
 * signed zeros added to the +0 the sum starts from, or the result is NaN
 * anyway (ONNXHostSequenceExtents.h), so the result does not change. */
#define UDONBLAS_ARENA_SGEMM_STRIDED_BATCHED(NAME, OPERAND_MASK)               \
  __kernel void NAME(__global const float *a, __global const float *b,         \
                     __global float *output, int m, int n, int k,              \
                     int stride_a, int stride_b, int stride_c,                 \
                     int batch_count, int transa, int transb, float alpha,     \
                     int k_begin) {                                            \
    size_t gid =                                                               \
        UDONBLAS_ARENA_Y() * UDONBLAS_ARENA_WIDTH + UDONBLAS_ARENA_X();        \
    size_t matrix_words = (size_t)m * (size_t)n;                               \
    size_t words = (size_t)batch_count * matrix_words;                         \
    float value = 0.0f;                                                        \
    if (gid < words) {                                                         \
      size_t batch = gid / matrix_words;                                       \
      size_t within = gid % matrix_words;                                      \
      int row = (int)(within / (size_t)n);                                     \
      int column = (int)(within % (size_t)n);                                  \
      size_t a_base = stride_a ? batch * (size_t)stride_a : 0u;                \
      size_t b_base = stride_b ? batch * (size_t)stride_b : 0u;                \
      (void)stride_c;                                                          \
      float sum = 0.0f;                                                        \
      for (int chunk = 0; chunk < UDONBLAS_ARENA_K_CHUNKS; ++chunk)            \
        if (chunk * UDONBLAS_ARENA_K_CHUNK < k &&                              \
            (chunk + 1) * UDONBLAS_ARENA_K_CHUNK > k_begin)                    \
          for (int lane = 0; lane < UDONBLAS_ARENA_K_CHUNK; ++lane) {          \
            int inner = chunk * UDONBLAS_ARENA_K_CHUNK + lane;                 \
            if (inner < k && inner >= k_begin) {                               \
              int a_index = transa ? inner * m + row : row * k + inner;        \
              int b_index = transb ? column * k + inner : inner * n + column;  \
              sum += a[(a_base + (size_t)a_index) & OPERAND_MASK] *            \
                     b[(b_base + (size_t)b_index) & OPERAND_MASK];             \
            }                                                                  \
          }                                                                    \
      value = alpha * sum;                                                     \
    }                                                                          \
    output[gid] = value;                                                       \
  }

/* Operands of up to 131072 words and a launch surface of 8192 x 2 words. */
UDONBLAS_ARENA_SGEMM_STRIDED_BATCHED(udonblas_arena_sgemm_strided_batched,
                                     131071u)
/* The same product for operands up to a whole Device buffer texture
 * (8192 x 8192 words) and a launch surface of 8192 x 16 words, for shapes the
 * kernel above cannot hold. Its taller surface charges every dispatch more
 * fragments, so the smaller kernel stays the default. */
UDONBLAS_ARENA_SGEMM_STRIDED_BATCHED(udonblas_arena_sgemm_strided_batched_large,
                                     67108863u)
