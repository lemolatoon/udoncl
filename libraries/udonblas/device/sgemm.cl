#ifndef __OPENCL_C_VERSION__
#include <stddef.h>
typedef unsigned int uint;
extern size_t udonblas_cpu_gid;
static inline size_t get_global_id(unsigned dimension) {
  return dimension == 0 ? udonblas_cpu_gid & 31u : udonblas_cpu_gid >> 5u;
}
#define __kernel
#define __global
#endif

#define UDONBLAS_WORDS 1024u
#define UDONBLAS_A_OFFSET 0u
#define UDONBLAS_B_OFFSET 256u
#define UDONBLAS_C_OFFSET 512u
#define UDONBLAS_RESULT_OFFSET 768u

/* One fragment owns one result element. The remaining fragments preserve the
 * packed state, which makes the kernel legal for UdonCL's ping-pong gather
 * profile and lets later BLAS operations share the same buffer ABI. */
__kernel void udonblas_sgemm(__global const float *input,
                             __global float *output, int m, int n, int k,
                             int transa, int transb, int lda, int ldb, int ldc,
                             float alpha, float beta) {
  uint gid = (uint)get_global_id(1) * 32u + (uint)get_global_id(0);
  float value = input[gid];
  if (gid >= UDONBLAS_RESULT_OFFSET &&
      gid < UDONBLAS_RESULT_OFFSET + (uint)(m * n)) {
    uint element = gid - UDONBLAS_RESULT_OFFSET;
    uint row = element % (uint)m;
    uint column = element / (uint)m;
    float sum = 0.0f;
    for (uint inner = 0; inner < 16u; ++inner) {
      uint a_index =
          transa == 0 ? row + inner * (uint)lda : inner + row * (uint)lda;
      uint b_index =
          transb == 0 ? inner + column * (uint)ldb : column + inner * (uint)ldb;
      if (inner < (uint)k)
        sum += input[UDONBLAS_A_OFFSET + (a_index & 255u)] *
               input[UDONBLAS_B_OFFSET + (b_index & 255u)];
    }
    value =
        alpha * sum +
        beta * input[UDONBLAS_C_OFFSET + ((row + column * (uint)ldc) & 255u)];
  }
  output[gid] = value;
}
