#ifdef __OPENCL_C_VERSION__
#define UDONBLAS_ARENA_SGEMV_X() ((size_t)get_global_id(0))
#define UDONBLAS_ARENA_SGEMV_Y() ((size_t)get_global_id(1))
#else
#include <stddef.h>
extern size_t udonblas_arena_sgemv_cpu_gid_x;
extern size_t udonblas_arena_sgemv_cpu_gid_y;
#define UDONBLAS_ARENA_SGEMV_X() (udonblas_arena_sgemv_cpu_gid_x)
#define UDONBLAS_ARENA_SGEMV_Y() (udonblas_arena_sgemv_cpu_gid_y)
#define __kernel
#define __global
#endif

#define UDONBLAS_ARENA_SGEMV_WIDTH 8192u
#define UDONBLAS_ARENA_SGEMV_CHUNK_WORDS 16u
#define UDONBLAS_ARENA_SGEMV_MASK 16383u

/* Private texture-arena backend for the public cuBLAS-shaped SGEMV contract.
 * ONNX M=1 Gemm reinterprets its row-major [K,N] right operand as a
 * column-major [N,K] matrix and dispatches OP_N. Product and reduction draws
 * keep each fragment bounded while preserving one logical SGEMV operation. */
__kernel void udonblas_arena_sgemv_product(__global const float *x,
                                           __global const float *a,
                                           __global float *output, int trans,
                                           int m, int n, int lda, int incx,
                                           float alpha) {
  size_t gid = UDONBLAS_ARENA_SGEMV_Y() * UDONBLAS_ARENA_SGEMV_WIDTH +
               UDONBLAS_ARENA_SGEMV_X();
  size_t output_length = trans == 0 ? (size_t)m : (size_t)n;
  size_t inner_length = trans == 0 ? (size_t)n : (size_t)m;
  size_t product_words = output_length * inner_length;
  float value = 0.0f;
  if (gid < product_words) {
    size_t row = gid / inner_length;
    size_t inner = gid % inner_length;
    size_t a_index =
        trans == 0 ? row + inner * (size_t)lda : inner + row * (size_t)lda;
    value = alpha * a[a_index & UDONBLAS_ARENA_SGEMV_MASK] *
            x[(inner * (size_t)incx) & UDONBLAS_ARENA_SGEMV_MASK];
  }
  output[gid] = value;
}

__kernel void udonblas_arena_sgemv_reduce(__global const float *products,
                                          __global float *output,
                                          int output_length, int inner_length) {
  size_t gid = UDONBLAS_ARENA_SGEMV_Y() * UDONBLAS_ARENA_SGEMV_WIDTH +
               UDONBLAS_ARENA_SGEMV_X();
  size_t chunks =
      ((size_t)inner_length + UDONBLAS_ARENA_SGEMV_CHUNK_WORDS - 1u) /
      UDONBLAS_ARENA_SGEMV_CHUNK_WORDS;
  float value = 0.0f;
  if (gid < (size_t)output_length * chunks) {
    size_t row = gid / chunks;
    size_t chunk = gid % chunks;
    size_t begin = chunk * UDONBLAS_ARENA_SGEMV_CHUNK_WORDS;
    float sum = 0.0f;
    for (size_t offset = 0; offset < UDONBLAS_ARENA_SGEMV_CHUNK_WORDS;
         ++offset) {
      size_t inner = begin + offset;
      if (inner < (size_t)inner_length)
        sum += products[(row * (size_t)inner_length + inner) &
                        UDONBLAS_ARENA_SGEMV_MASK];
    }
    value = sum;
  }
  output[gid] = value;
}

__kernel void udonblas_arena_sgemv_finalize(__global const float *partials,
                                            __global float *output,
                                            int output_length, int chunks) {
  size_t gid = UDONBLAS_ARENA_SGEMV_Y() * UDONBLAS_ARENA_SGEMV_WIDTH +
               UDONBLAS_ARENA_SGEMV_X();
  float value = 0.0f;
  if (gid < (size_t)output_length) {
    for (int chunk = 0; chunk < 16; ++chunk) {
      if (chunk < chunks)
        value += partials[(gid * (size_t)chunks + (size_t)chunk) &
                          UDONBLAS_ARENA_SGEMV_MASK];
    }
  }
  output[gid] = value;
}
