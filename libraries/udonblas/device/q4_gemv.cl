#ifdef __OPENCL_C_VERSION__
#define UDONBLAS_Q4_GID() ((size_t)get_global_id(0))
#else
#include <stddef.h>
typedef unsigned int uint;
extern size_t udonblas_cpu_gid;
#define UDONBLAS_Q4_GID() (udonblas_cpu_gid)
#define __kernel
#define __global
#endif

#define UDONBLAS_Q4_X_OFFSET 6144u
#define UDONBLAS_Q4_Y_OFFSET 6656u
#define UDONBLAS_Q4_RESULT_OFFSET 7168u

/* One fragment owns one output row. Packed Q4 bytes are represented as exact
 * numeric float values during the active tile, avoiding integer bitcasts while
 * This is a legacy internal tile layout; public Q4 storage belongs to UdonATen.
 */
__kernel void udonblas_sgemv_q4(__global const float *input,
                                __global float *output, int m, int k, int lda,
                                int incx, int incy, float alpha, float beta) {
  uint gid = (uint)UDONBLAS_Q4_GID();
  float value = input[gid];
  if (gid >= UDONBLAS_Q4_RESULT_OFFSET &&
      gid < UDONBLAS_Q4_RESULT_OFFSET + (uint)m) {
    uint row = gid - UDONBLAS_Q4_RESULT_OFFSET;
    float sum = 0.0f;
    for (uint inner = 0; inner < 128u; ++inner) {
      if (inner < (uint)k) {
        uint block = inner / 32u;
        uint in_block = inner % 32u;
        uint base = row * (uint)lda + block * 17u;
        float low = input[(base + in_block / 2u) & 8191u];
        float high = 0.0f;
        if (low >= 128.0f) {
          low -= 128.0f;
          high += 8.0f;
        }
        if (low >= 64.0f) {
          low -= 64.0f;
          high += 4.0f;
        }
        if (low >= 32.0f) {
          low -= 32.0f;
          high += 2.0f;
        }
        if (low >= 16.0f) {
          low -= 16.0f;
          high += 1.0f;
        }
        float nibble = (in_block % 2u) == 0u ? low : high;
        float quantized = nibble >= 8.0f ? nibble - 16.0f : nibble;
        sum += quantized * input[(base + 16u) & 8191u] *
               input[(UDONBLAS_Q4_X_OFFSET + inner * (uint)incx) & 8191u];
      }
    }
    value = alpha * sum +
            beta * input[(UDONBLAS_Q4_Y_OFFSET + row * (uint)incy) & 8191u];
  }
  output[gid] = value;
}
